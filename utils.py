from collections import defaultdict, deque
import torch
import time
import datetime
import torchvision


def create_lr_scheduler(optimizer,
                        num_step,
                        epochs,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):

    assert num_step > 0 and epochs > 0
    def f(x):
        if warmup and x <= num_step * warmup_epochs:
            alpha = x / (num_step * warmup_epochs)
            # -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # -> 0
            return (1 - (x - warmup_epochs * num_step) / (num_step * (epochs - warmup_epochs))) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

#多分类
class ConfusionMatrix2(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
    
    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            index = (a >= 0) & (a < n)
            m = n * a[index].long() + b[index].long()
            self.mat += torch.bincount(m, minlength=n ** 2).view(n, n)

    


    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def getCM(self):
        return self.mat
    

    #二分类
    def compute(self):
        m = self.mat

        #TP
        TP = m[1][1]
        #FN
        FN = m[1][0]
        #FP
        FP = m[0][1]

        #precision
        precision = TP / (TP + FP)
        #recall
        recall = TP / (TP + FN)
        #iou
        iou = TP / (TP + FP + FN)
        #dice
        dice = (2 * TP) / (2 * TP + FN + FP)
        #f1-score
        f1 = (2 * precision * recall) / (precision + recall)
        inds = [precision, recall, dice, f1, iou]
     
        return inds

    # 二分类
    def __str__(self):
        precision, recall, dice, f1, iou = self.compute()
        return (
                'precision: {}\n'
                'recall: {}\n'
                'dice: {}\n'
                'f1: {}\n'
                'iou: {}\n'
                ).format(
                    precision,
                    recall,
                    dice,
                    f1,
                    iou,
                )

class ConfusionMatrixM(object):
    def _init__(self, num_classes):
        super().__init__(num_classes)

    #多分类
    def compute(self):
        m = self.mat

        diag = m.diag()
        #全局预测准确率
        global_acc = diag.sum() / m.sum()
        sum0 = m.sum(0)
        sum1 = m.sum(1)
        #每个类别准确率
        acc = diag / sum1
        #TP
        TP = diag
        #FN
        FN = sum1 - diag
        #FP
        FP = sum0 - diag

        #precision
        precision = TP / (TP + FP)
        #recall
        recall = TP / (TP + FN)
        #iou
        iou = TP / (TP + FP + FN)
        #dice
        dice = (2 * TP) / (2 * TP + FN + FP)
        #f1-score
        f1 = (2 * precision * recall) / (precision + recall)
        mpre = precision.mean().item()
        mrecall = recall.mean().item()
        mf1 = f1.mean().item()
        miou = iou.mean() .item()
        mdice = dice.mean().item()
        inds = [acc, iou, precision, recall, f1, dice, global_acc.item(), miou, mpre, mrecall, mf1, mdice]
     
        return inds

    # 多分类
    def __str__(self):
        acc, iou, precision, recall, f1, dice, global_acc, miou, mprecision, mrecall, mf1, mdice = self.compute()
        return (
                'acc: {}\n'
                'iou: {}\n'
                'precision: {}\n'
                'recall: {}\n'
                'f1: {}\n'
                'dice: {}\n'
                'global_acc: {}\n'
                'miou: {}\n'
                'mprecison: {}\n'
                'mrecall: {}\n'
                'mf1: {}\n'
                'mdice: {}\n'
                ).format(
                    ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
                    ['{:.2f}'.format(i) for i in (iou * 100).tolist()],
                    ['{:.2f}'.format(i) for i in (precision * 100).tolist()],
                    ['{:.2f}'.format(i) for i in (recall * 100).tolist()],
                    ['{:.2f}'.format(i) for i in (f1 * 100).tolist()],
                    ['{:.2f}'.format(i) for i in (dice * 100).tolist()],
                    global_acc * 100,
                    miou * 100,
                    mprecision * 100,
                    mrecall * 100,
                    mf1 * 100,
                    mdice * 100
                )
    
