import torch
from tqdm import tqdm
from utils import ConfusionMatrix2

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, ce_loss, dice_loss):
    model.train()
    data_loader = tqdm(data_loader)
    total_loss = torch.zeros(1).to(device)
    for step, data in enumerate(data_loader):
        img, target = data[:, 2:3].repeat(1, 3, 1, 1).float(), data[:, 4].float()
        img, target = img.to(device), target.to(device)
        pred = model(img)
        loss_ce = ce_loss(pred, target.long())
        loss_dice = dice_loss(pred, target, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

        optimizer.zero_grad()
        loss.backward()

        if not torch.isfinite(loss):
            print('Loss is nan, exit...')
            exit(1)

        total_loss += loss.detach()
        optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]['lr']

        data_loader.desc = f'[epoch {epoch}] lr: {lr:6f} loss: {total_loss.item() / (step + 1):.3f}'


    return total_loss.item() / (step + 1), lr


def evaluate(model, data_loader, device, num_classes, epoch):
    model.eval()
    cm = ConfusionMatrix2(num_classes)
    data_loader = tqdm(data_loader)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            img, target = data[:, 2:3].repeat(1, 3, 1, 1).float(), data[:, 4].float()
            img, target = img.to(device).to(device), target.to(device)
            pred = model(img).argmax(dim=1)
            data_loader.desc = f'[val epoch {epoch}]'
            cm.update(target.flatten(), pred.flatten())
    return cm