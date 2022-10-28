from fileinput import filename
import os
import random
import numpy as np
from omegaconf import base
import torch
import argparse
import time
import logging
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from network import U_Net as Net
# from networks.vision_transformer import SwinUnet as Net
from net import Net
import torch.distributed as dist
from distributed_train_eval import train_one_epoch, evaluate
from utils import create_lr_scheduler, DiceLoss
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
from distributed_utils import setup, clean_up
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from ATLASDataset import ATLASDataset
import h5py

def args_parser():
    parser = argparse.ArgumentParser('train script.')
    parser.add_argument('--deterministic', default=True)
    parser.add_argument('--seed', default=22)
    parser.add_argument('--device', default='cuda', help='cpu or cuda')
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
    parser.add_argument('--sync_bn', default=True, type=bool)
    parser.add_argument('--world_size', default=2, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--rank', default=0)
    parser.add_argument('--backend', default='nccl')

    return parser.parse_args()

def get_logger(log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    format = "%(asctime)s - %(levelname)s - %(message)s"
    log_file = log_path + 'log.txt'
    date_fmt = "%Y-%m-%d %H:%M:%S %p"
    logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO, format=format, datefmt=date_fmt)
    return logging

def create_model(in_ch, out_ch, pretrain=False):
    model = Net(in_ch, out_ch)
    return model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    setup(args)
    rank = args.rank
    set_seed(args.seed + rank)
    if args.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    
    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    train = h5py.File(r"/home/infinity/Datasets/ATLAS/train")
    val = h5py.File(r"/home/infinity/Datasets/ATLAS/validation")
    test = h5py.File(r"/home/infinity/Datasets/ATLAS/test")

    train_dataset = ATLASDataset(train, 25892, True)
    # train_dataset = ATLASDataset(train, 8694, True)
    val_dataset = ATLASDataset(val, 8694, None)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])


    print('-------creating data loaders-------')
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    def worker_init_fn(worker_id):
        random.seed(args.seed + rank + worker_id)

    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                sampler=train_sampler,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True,
                                worker_init_fn=worker_init_fn)

    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                sampler=val_sampler,
                                num_workers=num_workers,
                                pin_memory=True)

    logger = None
    weight_path = './weight'
    if rank == 0:
        logger = get_logger(args.log_path)
        dir = './runs'
        if not os.path.exists(dir):
            os.mkdir(dir)
        tb = SummaryWriter(dir)
        
        if not os.path.exists(weight_path):
            os.mkdir(weight_path)
    
    device = torch.device(rank)
    
    model = create_model(3, num_classes).to(device)

    if args.sync_bn:
        print('-------use sync_bn-------')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model_without_ddp = model.module
    
    optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    
    dice_loss = DiceLoss(num_classes)
    ce_loss = CrossEntropyLoss()

    if len(args.resume):
        checkpoint = torch.load(weight_path + '/checkpoint_'+ args.resume + '_epoch.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    cnt = 0
    pre_dice = 0
    pre_f1 = 0
    pre_iou = 0

    print('-------start training-------')

    start_time = time.time()

    for epoch in range(args.start_epoch - 1, args.epochs):
        epoch = epoch + 1
        train_sampler.set_epoch(epoch)
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, lr_scheduler, ce_loss, dice_loss)
        if rank == 0:
            tb.add_scalar('mean_loss', mean_loss, epoch)
            tb.add_scalar('lr', lr, epoch)
            msg = f'[epoch: {epoch}]\ttrain_loss: {mean_loss:.3f}\tlr: {lr:.6f}'
            logger.info(msg)
        if epoch >= 50:
            cm = evaluate(model, val_loader, device, num_classes, epoch)
            inds = cm.compute()
            pre, recall, dice, f1, iou = inds[:]
            if iou > pre_iou or f1 > pre_f1 or dice > pre_dice or epoch % 5 == 0:
                pre_iou = pre_iou if pre_iou > iou else iou
                pre_f1 = pre_f1 if pre_f1 > f1 else f1
                pre_dice = pre_dice if pre_dice > dice else dice
                if iou == pre_iou or f1 == pre_f1 or dice == pre_dice: cnt = 0
                if rank == 0:
                    to_save = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'cm': cm.getCM()
                        }
                    print(f'\tsaving checkpoint_{epoch}_epoch.pth.')
                    torch.save(to_save, weight_path + '/' + f'checkpoint_{epoch}_epoch.pth')
            if rank == 0:
                msg = f'\tval:\tpre: {pre}\trecall: {recall}\tdice: {dice}\tf1: {f1}\tiou: {iou}'
                logger.info(msg)
                # logger.info(str(cm))
                msg = f'\t\t{str(cm.getCM())}'
                print('\t',cm.getCM())
                logger.info(msg)
                tb.add_scalar('pre', pre, epoch)
                tb.add_scalar('recall', recall, epoch)
                tb.add_scalar('dice', dice, epoch)
                tb.add_scalar('f1', f1, epoch)
                tb.add_scalar('iou', iou, epoch)
               
        dist.barrier()
        cnt = cnt + 1
        if cnt > 50:
            print("evaluating indicators have not changed since 50 epochs ago, stop trainning!")
            break;
    
    clean_up()
    if rank == 0:
        tb.close()
    print(f'Finish training, tiem consuming {time.time() - start_time}.')
  
if __name__ == '__main__':
    args = args_parser()
    main(args)
