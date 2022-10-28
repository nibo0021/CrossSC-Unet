import os
import torch
import torch.distributed as dist





def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup(args):
    assert 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    args.rank = eval(os.environ['RANK'])
    args.world_size = eval(os.environ['WORLD_SIZE'])
    # os.environ['OPM_NUM_THREADS'] = '1'

    # print(f'RANK/WORLD_SIZE: {args.rank}/{args.world_size}.')

    dist.init_process_group(backend=args.backend, 
                            init_method=args.dist_url, 
                            rank= args.rank,
                            world_size=args.world_size)
    setup_for_distributed(args.rank == 0)


def clean_up():
    dist.destroy_process_group()

#多分类
class ConfusionMatrix(object):
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
    #多分类
    # def compute(self):
    #     m = self.mat

    #     diag = m.diag()
    #     #全局预测准确率
    #     global_acc = diag.sum() / m.sum()
    #     sum0 = m.sum(0)
    #     sum1 = m.sum(1)
    #     #每个类别准确率
    #     acc = diag / sum1
    #     #TP
    #     TP = diag
    #     #FN
    #     FN = sum1 - diag
    #     #FP
    #     FP = sum0 - diag

    #     #precision
    #     precision = TP / (TP + FP)
    #     #recall
    #     recall = TP / (TP + FN)
    #     #iou
    #     iou = TP / (TP + FP + FN)
    #     #dice
    #     dice = (2 * TP) / (2 * TP + FN + FP)
    #     #f1-score
    #     f1 = (2 * precision * recall) / (precision + recall)
    #     mpre = precision.mean().item()
    #     mrecall = recall.mean().item()
    #     mf1 = f1.mean().item()
    #     miou = iou.mean() .item()
    #     mdice = dice.mean().item()
    #     inds = [acc, iou, precision, recall, f1, dice, global_acc.item(), miou, mpre, mrecall, mf1, mdice]
     
    #     return inds
    


    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def getCM(self):
        return self.mat

    def reduce_from_all_processes(self):
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.mat)
    # 多分类
    # def __str__(self):
    #     acc, iou, precision, recall, f1, dice, global_acc, miou, mprecision, mrecall, mf1, mdice = self.compute()
    #     return (
    #             'acc: {}\n'
    #             'iou: {}\n'
    #             'precision: {}\n'
    #             'recall: {}\n'
    #             'f1: {}\n'
    #             'dice: {}\n'
    #             'global_acc: {}\n'
    #             'miou: {}\n'
    #             'mprecison: {}\n'
    #             'mrecall: {}\n'
    #             'mf1: {}\n'
    #             'mdice: {}\n'
    #             ).format(
    #                 ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
    #                 ['{:.2f}'.format(i) for i in (iou * 100).tolist()],
    #                 ['{:.2f}'.format(i) for i in (precision * 100).tolist()],
    #                 ['{:.2f}'.format(i) for i in (recall * 100).tolist()],
    #                 ['{:.2f}'.format(i) for i in (f1 * 100).tolist()],
    #                 ['{:.2f}'.format(i) for i in (dice * 100).tolist()],
    #                 global_acc * 100,
    #                 miou * 100,
    #                 mprecision * 100,
    #                 mrecall * 100,
    #                 mf1 * 100,
    #                 mdice * 100
    #             )

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
