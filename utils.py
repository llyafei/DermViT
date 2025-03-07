import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from itertools import cycle
from sklearn.metrics import roc_auc_score, roc_curve, auc, recall_score, precision_score, classification_report, \
    f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from warmup_scheduler import GradualWarmupScheduler
from warmup_scheduler_pytorch import WarmUpScheduler
import matplotlib.pyplot as plt
from thop import clever_format, profile
from prettytable import PrettyTable
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

    def forward(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, device, class_num=7, gamma=2, alpha=None, reduction='mean', ):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num
        self.device = device

    def forward(self, logits, labels):
        alpha = self.alpha[labels]  # shape=(bs)
        log_softmax = torch.log_softmax(logits, dim=1)  # shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=labels.view(-1, 1))  # shape=(bs, 1)
        logpt = logpt.view(-1)  # shape=(bs)
        ce_loss = -logpt  # log_softmax
        pt = torch.exp(logpt)  # log_softmax and exp,shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # focal loss,shape=(bs)
        if self.reduction == "mean":
            return focal_loss.sum() / alpha.sum()
        if self.reduction == "sum":
            return torch.sum(focal_loss)


class reduced_focal_loss(torch.nn.Module):
    """Reduced Focal Loss: https://arxiv.org/pdf/1903.01347v2.pdf"""

    def __init__(self, device, threshold, gamma=2, reduction='mean', ):
        super(reduced_focal_loss, self).__init__()
        self.gamma = gamma
        self.threshold = threshold
        self.reduction = reduction
        self.device = device

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels.to(self.device), reduction="none")
        log_pt = -ce_loss
        pt = torch.exp(log_pt)

        low_th_weight = torch.ones_like(pt)
        high_th_weight = (1 - pt) ** self.gamma / (self.threshold ** self.gamma)
        weights = torch.where(pt < self.threshold, low_th_weight, high_th_weight)

        rfl = weights * ce_loss

        if self.reduction == "sum":
            rfl = rfl.sum()
        elif self.reduction == "mean":
            rfl = rfl.mean()
        else:
            raise ValueError(f"reduction '{self.reduction}' is not valid")
        return rfl


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-2):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def read_split_data(root: str, val_rate: float = 0.15):
    random.seed(17)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    skin_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    skin_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(skin_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".JPEG"]
    # Go through the files under each folder
    for cla in skin_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        # Go through each type of picture and divide the training set and verification set according to proportion
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) >= 0, "number of training images must greater than 0."
    assert len(val_images_path) >= 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        plt.bar(range(len(skin_class)), every_class_num, align='center')
        plt.xticks(range(len(skin_class)), skin_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def read_split_data_2018(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    skin_class_train = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]

    skin_class_train.sort()

    class_indices = dict((k, v) for v, k in enumerate(skin_class_train))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices_2018.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    train_class_num = []
    val_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".JPEG"]

    for cla in skin_class_train:
        cla_path = os.path.join(train_dir, cla)

        images = [os.path.join(train_dir, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        images.sort()

        image_class = class_indices[cla]

        train_class_num.append(len(images))


        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    for cla in skin_class_train:
        cla_path = os.path.join(val_dir, cla)

        images = [os.path.join(val_dir, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        images.sort()

        image_class = class_indices[cla]

        val_class_num.append(len(images))

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images were found in the dataset.".format(len(train_images_path) + len(val_images_path)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label


def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion,scheduler=None):
    model.train()
    loss_function = criterion
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):

        images, labels = data

        sample_num += images.shape[0]

        # pred.shape:(batch_size, num_classes)
        pred = model(images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    # return accu_loss.item() / (step + 1), accu_num.item() / sample_num
    return accu_loss.item() / (len(data_loader)), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, average_way, criterion=None):
    model.eval()
    loss_function = nn.CrossEntropyLoss() if criterion is None else criterion

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    LOSS = []
    PROBS = []
    LABELS = []
    PREDICTS = []
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        logits = model(images.to(device))
        probs = logits.softmax(1)
        # [batch_size, num_classes]
        pred_classes = torch.max(logits, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(logits, labels.to(device))
        accu_loss += loss

        LOSS.append(loss.detach().cpu().numpy())
        PROBS.append(probs.detach().cpu())
        LABELS.append(labels.detach().cpu())
        PREDICTS.append(pred_classes.detach().cpu())

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num, )
    # numpy
    PROBS = torch.cat(PROBS).numpy()
    LABELS = torch.cat(LABELS).numpy()
    PREDICTS = torch.cat(PREDICTS).numpy()
    # auc,prec
    AUC, prec, recall, f1 = get_metrics(y_true=LABELS, y_pred=PREDICTS, y_probs=PROBS, average_way=average_way)
    print("[valid epoch {}] macro_auc: {:.3f}, micro_auc: {:.3f}".format(epoch, AUC['macro'], AUC['micro']))
    if epoch > 20:
        print('---------------classification_report---------------')
        print(classification_report(LABELS, PREDICTS, target_names=['C1', 'C2', 'C3', 'C4', 'C5', 'C6',
                                                                    'C7','C8']))  # 'C4', 'C5', 'C6', 'C7','C8'
        # draw_confision(LABELS, PREDICTS)
        # ROC
        # get_roc_auc(LABELS, PROBS, draw_figure=True)
    return accu_loss.item() / (len(data_loader)), accu_num.item() / sample_num, AUC, prec, recall, f1


def draw_confision(LABELS, PREDICTS):
    conf_matrix = confusion_matrix(LABELS, PREDICTS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'],
                yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def get_roc_auc(true_labels, probs, num_classes=7, draw_figure: bool = False):
    binary_labels = label_binarize(true_labels, classes=[0, 1, 2, 3, 4, 5, 6])

    # ROC
    fpr = dict()
    tpr = dict()
    auc_score = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(binary_labels[:, i], probs[:, i])
        auc_score[i] = auc(fpr[i], tpr[i])

    # macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auc_score["macro"] = auc(fpr["macro"], tpr["macro"])

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), probs.ravel())
    auc_score["micro"] = auc(fpr["micro"], tpr["micro"])

    # draw auc
    if draw_figure:
        plt.figure(figsize=(10, 6))
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {auc_score[i]:.2f})')
        lw = 2
        plt.figure()
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(auc_score["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(auc_score["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, auc_score[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    return auc_score


def get_metrics(y_true, y_pred, y_probs, average_way):
    LABELS = y_true
    PREDICTS = y_pred
    PROBS = y_probs
    AUC = dict()
    PREC = dict()
    RECALL = dict()
    F1SCORE = dict()
    # LABELS -> 为one-hot
    AUC['macro'] = roc_auc_score(LABELS, PROBS, multi_class='ovr', average='macro')
    AUC['micro'] = roc_auc_score(LABELS, PROBS, multi_class='ovo')
    # precision
    PREC['macro'] = precision_score(y_true=LABELS, y_pred=PREDICTS, average='macro')
    PREC['micro'] = precision_score(y_true=LABELS, y_pred=PREDICTS, average='micro')
    # recall
    RECALL['macro'] = recall_score(y_true=LABELS, y_pred=PREDICTS, average='macro')
    RECALL['micro'] = recall_score(y_true=LABELS, y_pred=PREDICTS, average='micro')
    # F1-score
    F1SCORE['macro'] = f1_score(y_true=LABELS, y_pred=PREDICTS, average='macro')
    F1SCORE['micro'] = f1_score(y_true=LABELS, y_pred=PREDICTS, average='micro')

    return AUC, PREC[average_way], RECALL[average_way], F1SCORE[average_way]


class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert('RGB')
            # raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # Normalize
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def calculate_flops(model, x, print_table: bool = True):
    # FLOPs=2MACs
    macs, params = profile(model, inputs=(x,))
    macs, flops, params = clever_format([macs, 2 * macs, params], "%.3f")
    table = PrettyTable()
    table.field_names = ["MACs", "FLOPs", "params"]
    table.add_row([macs, flops, params])
    if print_table:
        print(table)


def same_seeds(seed):
    # fix random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def plot_class_preds(model,
                     images_dir: str,
                     transform,
                     num_plot: int = 7,
                     device="cpu",
                     label_dir: str = None,
                     show: bool = False):
    if not os.path.exists(images_dir):
        print("not found {} path, ignore add figure.".format(images_dir))
        return None
    if label_dir is None:
        label_path = os.path.join(images_dir, "label.txt")
    else:
        label_path = label_dir
    if not os.path.exists(label_path):
        print("not found {} file, ignore add figure".format(label_path))
        return None

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "not found {}".format(json_label_path)
    json_file = open(json_label_path, 'r')
    # {"0": "daisy"}
    flower_class = json.load(json_file)
    # {"daisy": "0"}
    class_indices = dict((v, k) for k, v in flower_class.items())

    # reading label.txt file
    label_info = []
    with open(label_path, "r") as rd:
        for line in rd.readlines():
            line = line.strip()
            if len(line) > 0:
                split_info = [i for i in line.split(" ") if len(i) > 0]
                assert len(split_info) == 2, "label format error, expect file_name and class_name"
                image_name, class_name = split_info
                image_path = os.path.join(images_dir, image_name)

                if not os.path.exists(image_path):
                    print("not found {}, skip.".format(image_path))
                    continue

                if class_name not in class_indices.keys():
                    print("unrecognized category {}, skip".format(class_name))
                    continue
                label_info.append([image_path, class_name])

    if len(label_info) == 0:
        return None

    # get first num_plot info
    if len(label_info) > num_plot:
        label_info = label_info[:num_plot]

    num_imgs = len(label_info)
    images = []
    labels = []
    for img_path, class_name in label_info:
        # read img
        img = Image.open(img_path).convert("RGB")
        label_index = int(class_indices[class_name])

        # preprocessing
        img = transform(img)
        images.append(img)
        labels.append(label_index)

    # batching images
    images = torch.stack(images, dim=0).to(device)

    # inference
    with torch.no_grad():
        output = model(images)
        probs, preds = torch.max(torch.softmax(output, dim=1), dim=1)
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()

    # width, height
    fig = plt.figure(figsize=(num_imgs * 2.5, 3), dpi=100)
    for i in range(num_imgs):

        ax = fig.add_subplot(1, num_imgs, i + 1, xticks=[], yticks=[])

        # CHW -> HWC
        npimg = images[i].cpu().numpy().transpose(1, 2, 0)

        # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
        npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        # numpy
        plt.imshow(npimg.astype('uint8'))

        title = "(predict: {}), {:.2f}%\n(label: {})".format(
            flower_class[str(preds[i])],  # predict class
            probs[i] * 100,  # predict probability
            flower_class[str(labels[i])]  # true class
        )
        ax.set_title(title, color=("green" if preds[i] == labels[i] else "red"))

    if show:
        plt.show()
    return fig
