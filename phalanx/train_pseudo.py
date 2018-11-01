import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fastprogress import master_bar, progress_bar

from lovasz_losses import lovasz_hinge
from utils import do_kaggle_metric
from salt_dataset import SaltDataset, trainImageFetch, semi_trainImageFetch
from unet_model import Res34Unetv3, Res34Unetv4, Res34Unetv5

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='res34v5', type=str, help='Model version')
parser.add_argument('--fine_size', default=101, type=int, help='Resized image size')
parser.add_argument('--pad_left', default=13, type=int, help='Left padding size')
parser.add_argument('--pad_right', default=14, type=int, help='Right padding size')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--epoch', default=300, type=int, help='Number of training epochs')
parser.add_argument('--snapshot', default=5, type=int, help='Number of snapshots per fold')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--save_weight', default='weights/stage3/', type=str, help='weight save space')
parser.add_argument('--max_lr', '--learning-rate', default=0.01, type=float, help='max learning rate')
parser.add_argument('--min_lr', '--learning-rate', default=0.001, type=float, help='min learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--pseudo_path', default='../data/pseudolabels_v2/', type=float, help='pseudo labels path')

args = parser.parse_args()
image_size = args.fine_size + args.pad_left + args.pad_right
args.weight_name = 'model_' + str(image_size) + '_' + args.model

if not os.path.isdir(args.save_weight):
    os.mkdir(args.save_weight)

device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')


# Validation function
def test(test_loader, model):
    running_loss = 0.0
    predicts = []
    truths = []

    model.eval()
    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs[:, :, args.pad_left:args.pad_left + args.fine_size,
                      args.pad_left:args.pad_left + args.fine_size].contiguous()
            loss = lovasz_hinge(outputs.squeeze(1), masks.squeeze(1))

        predicts.append(nn.Sigmoid()(outputs).detach().cpu().numpy())
        truths.append(masks.detach().cpu().numpy())
        running_loss += loss.item() * inputs.size(0)

    predicts = np.concatenate(predicts).squeeze()
    truths = np.concatenate(truths).squeeze()
    precision, _, _ = do_kaggle_metric(predicts, truths, 0.52)
    precision = precision.mean()
    epoch_loss = running_loss / val_data.__len__()
    return epoch_loss, precision


# Training function
def train(train_loader, model):
    running_loss = 0.0
    data_size = train_data.__len__()

    model.train()
    for inputs, masks, labels in progress_bar(train_loader, parent=mb):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            logit = model(inputs)
            loss = lovasz_hinge(logit.squeeze(1), masks.squeeze(1))
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        mb.child.comment = 'loss: {}'.format(loss.item())
    epoch_loss = running_loss / data_size
    return epoch_loss


if __name__ == '__main__':
    best_acc = 0  # validation accuracy

    # Setup data
    val_id = np.load('../data/valid_id.npy')
    image_val, mask_val = trainImageFetch(val_id)
    image_train, mask_train = semi_trainImageFetch(args.pseudo_path)

    train_data = SaltDataset(image_train, mask_train, is_train=True, fine_size=args.fine_size, pad_left=args.pad_left,
                             pad_right=args.pad_right)
    val_data = SaltDataset(image_val, mask_val, is_val=True, fine_size=args.fine_size, pad_left=args.pad_left,
                           pad_right=args.pad_right)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Get model
    if args.model == 'res34v3':
        salt = Res34Unetv3()
    elif args.model == 'res34v4':
        salt = Res34Unetv4()
    elif args.model == 'res34v5':
        salt = Res34Unetv5()
    salt = salt.to(device)

    # Setup optimizer and scheduler
    scheduler_step = args.epoch // args.snapshot
    optimizer = torch.optim.SGD(salt.parameters(), lr=args.max_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, args.min_lr)

    # Start train
    mb = master_bar(range(args.epoch))
    for epoch in mb:
        train_loss = train(train_loader, salt)
        val_loss, precision = test(val_loader, salt)
        lr_scheduler.step()

        if (epoch + 1) % scheduler_step == 0:
            optimizer = torch.optim.SGD(salt.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, args.min_lr)
            best_param = salt.state_dict()

        if precision > best_acc:
            best_acc = precision

        # print train/val loss and val accuracy
        mb.write(
            'epoch: {} train_loss: {:.3f} test_loss: {:.3f} accuracy: {:.3f}'.format(epoch + 1, train_loss, val_loss,
                                                                                     precision))

    # save best accuracy weight
    torch.save(best_param, args.save_weight + args.weight_name + '.pth')