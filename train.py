import os
import math
import torchvision
from argparse import ArgumentParser
import torch
import argparse
import utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from DermViT import DermViT_base as Create_MyModel
# from resnet import resnet101 as Create_MyModel
# from vit import vit_base_patch16_224 as Create_MyModel
# from CAT_Net import CAT_Net_normal as Create_MyModel
# from contrast_model.ConvNeXt import convnext_tiny as Create_MyModel
from utils import train_one_epoch, evaluate, read_split_data, MyDataSet, calculate_flops, same_seeds, plot_class_preds, \
    FocalLoss, MultiCEFocalLoss, reduced_focal_loss
# from visualizations import plot_grad_cam_visualization
import warnings
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    same_seeds(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # number of workers
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    tb_writer = SummaryWriter(log_dir=args.runs_dir)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(degrees=(0, 360)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../deep-learning-for-image-processing/data_set"))
    train_dir = os.path.join(data_root, args.train_dir)
    test_dir = os.path.join(data_root, args.test_dir)
    assert os.path.exists(train_dir), "{} path does not exist.".format(train_dir)

    # # train dataset
    # train_dataset = ImageFolder(root=train_dir,
    #                             transform=data_transform["train"])
    # # val dataset
    # val_dataset = ImageFolder(root=test_dir,
    #                           transform=data_transform["val"])
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(train_dir, val_rate=0.15)

    # train dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    # val dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # training loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
    # 验证加载器
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw)

    # model
    model = Create_MyModel(num_classes=args.num_classes, drop_path_rate=args.drop_path_rate_bra,
                           drop_path_rate_CGLU=args.drop_path_rate_CGLU)
    # model = Create_MyModel(num_classes=args.num_classes)
    # pretrained=args.pretrained

    # FLOPs and MACs
    print(args.model + ":")
    calculate_flops(model, x=torch.randn(1, 3, args.image_size, args.image_size), print_table=True)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    scheduler = None
    if args.optim == "Adam":
        # optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        optimizer = optim.Adam([{'params': params, 'initial_lr': args.lr}], lr=args.lr)
    elif args.optim == "SGD":
        optimizer = optim.SGD([{'params': params, 'initial_lr': args.lr}], lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == "AdamW":
        # optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        optimizer = optim.AdamW([{'params': params, 'initial_lr': args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    # scheduler
    if args.scheduler == "LambdaLR":
        # new_lr=lr×lambda_function(epoch)
        lf_func = lambda epoch: ((1 + math.cos(epoch * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
        # lf_func = lambda epoch:0.1 if epoch % 50 == 0 else 1.0
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf_func, last_epoch=args.last_epoch)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=args.last_epoch, verbose=True)
    if args.scheduler == "WarmUp":
        scheduler = utils.create_lr_scheduler(optimizer,
                                              len(train_loader),
                                              args.epochs,
                                              warmup=True,
                                              warmup_epochs=10,
                                              warmup_factor=1e-3,
                                              end_factor=args.lrf)
    # checkpoint
    if args.load_from_checkpoint and len(args.checkpoint_dir) > 0:
        checkpoint = torch.load("checkpoint/" + args.checkpoint_dir)
        print("load checkpoint from {}".format("checkpoint/" + args.checkpoint_dir))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.last_epoch = checkpoint['epoch']
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    model.to(device)

    for name, para in model.named_parameters():
        if para.requires_grad:
            print("training {}".format(name))

    # TensorBoard
    tb_writer.add_text("super_paras", str(args))
    tb_writer.add_graph(model, torch.randn(1, 3, args.image_size, args.image_size).to(device))

    # train and val
    best_val_acc = 0
    best_epoch = 0
    if args.loss_func == 'ce':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.class_weight, dtype=torch.float32).to(device))
    elif args.loss_func == 'fl':
        criterion = FocalLoss(alpha=args.FL_alpha, gamma=args.FL_gamma, device=device)
    elif args.loss_func == 'mfl':
        criterion = MultiCEFocalLoss(gamma=args.FL_gamma, alpha=args.class_weight,
                                     device=device)
    else:
        criterion = reduced_focal_loss(gamma=args.FL_gamma, device=device, threshold=args.threshold)
    for epoch in range(args.last_epoch + 1, args.epochs):
        # train
        print(optimizer.param_groups[0]['lr'])
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                criterion=criterion,
                                                scheduler=None
                                                )

        scheduler.step()
        # validate
        val_loss, val_acc, AUC, prec, recall, f1 = evaluate(model=model,
                                                            data_loader=val_loader,
                                                            device=device,
                                                            epoch=epoch,
                                                            average_way=args.average_way
                                                            )

        # training log
        tb_writer.add_scalars("ALL/acc", {"train": train_acc, "val": val_acc}, epoch)
        tb_writer.add_scalars("ALL/loss", {"train": train_loss, "val": val_loss}, epoch)
        tb_writer.add_scalar("ALL/lr", optimizer.param_groups[0]["lr"], epoch)

        tb_writer.add_scalar("AUC/macro", AUC['macro'], epoch)
        tb_writer.add_scalar("AUC/micro", AUC['micro'], epoch)

        tb_writer.add_scalar("METRIC/prec", prec, epoch)
        tb_writer.add_scalar("METRIC/recall", recall, epoch)
        tb_writer.add_scalar("METRIC/f1", f1, epoch)

        # save model
        # torch.save(model.state_dict(), "./weights/{}-{}.pth".format(args.model, epoch))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_images_path': train_images_path,
                'train_images_label': train_images_label,
                'val_images_path': val_images_path,
                'val_images_label': val_images_label
            }, "./checkpoint/{}-{}".format(args.model, epoch))

        # add figure into tensorboard
        # fig = plot_class_preds(model=model,
        #                        images_dir="./pred_images",
        #                        transform=data_transform["val"],
        #                        num_plot=7,
        #                        device=device)
        # if fig is not None:
        #     if epoch > 20 and epoch % 10 == 0:
        #         tb_writer.add_figure("predictions vs. actuals",
        #                              figure=fig,
        #                              global_step=epoch)
    # save best
    best_model_path = args.runs_dir + "/{}_best-{}".format(args.model, best_epoch)
    checkpoint_best = torch.load("./checkpoint/{}-{}".format(args.model, best_epoch))
    torch.save(model.state_dict(), best_model_path)
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': checkpoint_best['model_state_dict'],
        'optimizer_state_dict': checkpoint_best['optimizer_state_dict'],
        'train_images_path': train_images_path,
        'train_images_label': train_images_label,
        'val_images_path': val_images_path,
        'val_images_label': val_images_label
    }, best_model_path)
    # visualization
    # visualization = plot_grad_cam_visualization(model,
    #                                             best_model_path,
    #                                             [model.layer4[-1]],
    #                                             "./pred_images")
    # tb_writer.add_figure("visualization", figure=visualization, global_step=best_epoch)


if __name__ == '__main__':
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="DermViT(+DFG+DCP)(ISIC2019)")
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=224)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--FL_alpha', type=float, default=1)
    parser.add_argument('--FL_gamma', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--class_weight', type=list, default=[1, 1, 1, 2, 1, 1, 1, 1])
    parser.add_argument('--loss_func', type=str, default='ce', choices=['fl', 'ce', 'mutiFl', 'rfl'])

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lrf', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=0.0002)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--drop_path_rate_bra', type=float, default=0.2)
    parser.add_argument('--drop_path_rate_CGLU', type=float, default=0.)
    parser.add_argument('--optim', default='AdamW', choices=['SGD', 'Adam', "AdamW"])
    parser.add_argument('--scheduler', default='LambdaLR', choices=['LambdaLR', 'WarmUp'])

    parser.add_argument('--average_way', default='macro', choices=['macro', 'micro'])

    parser.add_argument('--load_from_checkpoint', default=False)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='')
    parser.add_argument('--last_epoch', type=int, default=-1)

    parser.add_argument('--train_dir', type=str, default="ISIC2019/ISIC_2019_all_9classes")
    parser.add_argument('--test_dir', type=str, default="ISIC2019/test")
    parser.add_argument('--val_rate', type=float, default=0.15)
    parser.add_argument("--runs_dir", type=str, default="./runs/DermViT(+DFG+DCP)(ISIC2019)")
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', type=int, default=17)
    opt = parser.parse_args()

    print(opt)
    main(opt)
