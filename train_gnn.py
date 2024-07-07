import os
import math
import argparse
import logging
import torch
import torch.optim as optim
import timm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
from config import TrainConfig
from gnn import create_model
from utils import read_data, train_one_epoch, evaluate, TGNNDataSet

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    current_time = datetime.now().strftime("%Y-%m-%d")
    log_file_path = f"./logs/{current_time}.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    train_images_path, train_images_label = read_data(args.train_path)
    test_images_path, test_images_label = read_data(args.test_path)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.951,0.951, 0.951], [0.19, 0.19, 0.19])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.949,0.949,0.949],[0.196,0.196,0.196])])}

    # 实例化训练数据集
    train_dataset = TGNNDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = TGNNDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = create_model().to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "classifier" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.99))
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_acc = 0.
    early_stop_count = 0.
    for epoch in range(1, args.epochs+1):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()

        # validate
        val_loss, val_acc, precision, recall, f1 = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        logging.info("acc: {}, preccision: {}, recall: {}, f1: {}".format(round(val_acc, 3), round(precision, 3),
                                                                          round(recall, 3), round(f1, 3)))
        if best_acc < val_acc:
            if not os.path.isdir("./weight"):
                os.mkdir("./weight")
            torch.save(model.state_dict(), "./weight/best_model.pth")
            print("Saved epoch{} as new best model".format(epoch))
            best_acc = val_acc
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count == 10:
            break

        logging.info("epoch:  {}\tbest_acc:  {}\tearly_stop_count: {}".format(
            epoch, best_acc, early_stop_count))

    torch.save(model.state_dict(), "./weight/last_model.pth")

if __name__ == '__main__':
    main(args=TrainConfig())