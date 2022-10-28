import os
import torch
import argparse
import time
import logging
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from train_eval import train_one_epoch, evaluate
from net import Net
from utils import create_lr_scheduler, DiceLoss
from ATLASDataset import ATLASDataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
# from Models import NestedUNet as Net
from torch.nn.modules.loss import CrossEntropyLoss
import h5py

def args_parser():
    parser = argparse.ArgumentParser('train script.')
    # parser.add_argument()
    parser.add_argument('--device', default='cuda:0', help='cpu or cuda')
    parser.add_argument('--seed', default=22)
    parser.add_argument('--deterministic', default=True)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--log_path', default='./log/')
    parser.add_argument('--data_path', default='', help='dir of dataset')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--resume', default='')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--print_freq', default=10)

    return parser.parse_args()

# def get_logger(log_path):
#     if not os.path.exists(log_path):
#         os.mkdir(log_path)
#     format = "%(asctime)s - %(levelname)s - %(message)s"
#     log_file = log_path + 'log.txt'
#     date_fmt = "%Y-%m-%d %H:%M:%S %p"
#     logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO, format=format, datefmt=date_fmt)
#     return logging


def create_model(in_ch, out_ch, pretrain=False):
    model = Net(in_ch, out_ch)
    return model

# def set_seed(seed):

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)

def main(args):
    # if args.deterministic:
    #     cudnn.deterministic = True
    #     cudnn.benchmark = False
    # set_seed(args.seed)
    device = torch.device(args.device)
    # batch_size = args.batch_size
    num_classes = args.num_classes + 1
    # logger = get_logger(args.log_path)

    # train = h5py.File(r"/home/infinity/Datasets/ATLAS/train")
    # val = h5py.File(r"/home/infinity/Datasets/ATLAS/validation")
    test = h5py.File(r"/home/infinity/Datasets/ATLAS/test")

    # train_dataset = ATLASDataset(train, 25892, True)
    # train_dataset = ATLASDataset(train, 8694, True)
    # val_dataset = ATLASDataset(val, 8694, None)
    test_dataset = ATLASDataset(test, 8694, None)

    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)
    # train_loader = DataLoader(train_dataset,
    #                          batch_size=batch_size,
    #                          shuffle=True,
    #                          num_workers=num_workers,
    #                          pin_memory=True,
    #                          worker_init_fn=worker_init_fn)

    # val_loader = DataLoader(val_dataset,
    #                          batch_size=1,
    #                          num_workers=num_workers,
    #                          pin_memory=True,
    #                         )
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=1,
                             pin_memory=True,
                            )

    weight_path = './weight'
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)

    model = create_model(3, num_classes).to(device)

    # dice_loss = DiceLoss(num_classes)
    # ce_loss = CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    
    # if len(args.resume):
    #     checkpoint = torch.load(weight_path + '/checkpoint_'+ args.resume + '_epoch.pth', map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     args.start_epoch = checkpoint['epoch'] + 1

    checkpoint = torch.load(weight_path + '/checkpoint_'+ args.resume + '_epoch.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])  
    start_time = time.time()
    print('-------start testing-------')
    start_time = time.time()
    
    cm = evaluate(model, test_loader, device, num_classes, args.resume)
    inds = cm.compute()
    pre, recall, dice, f1, iou = inds[:]
    msg = f'\tval:\tpre: {pre}\trecall: {recall}\tdice: {dice}\tf1: {f1}\tiou: {iou}'
    print(msg)
    print(cm.getCM())
    print(f'Finish testing, tiem consuming {time.time() - start_time}.')
  
if __name__ == '__main__':
    args = args_parser()
    main(args)