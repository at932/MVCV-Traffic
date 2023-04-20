import numpy as np
import torch

"""
参考：
https://github.com/Cai-Yichao/torch_backbones/blob/master/utils/earlystopping.py
"""

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, savepath,patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.（距离上次val loss提升最大可容忍的epochs（否则将停止））
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.（是否打印val loss提升的信息）
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.（loss认为是提升时的loss的最小变化值）
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0 #验证loss开始增大的次数/回数
        self.best_score = None #记录最好的结果
        self.early_stop = False #是否停止的标志
        self.val_loss_min = np.Inf #最小loss
        self.delta = delta
        self.savepath = savepath

    def __call__(self, val_loss, model):

        score = -val_loss

        save_flag = False
        if self.best_score is None:

            #第一次迭代时候
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            save_flag = True
        elif score < self.best_score + self.delta:
            #（-cur_val_loss < -best_val_loss，即cur_val_loss>best_val_loss，验证集loss增大了，出现过拟合状况）
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True #val loss增大的次数大于容忍次数阈值，则停止
        else:
            #否则更新最好结果记录，同时保存结果
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            save_flag = True
            self.counter = 0
        return save_flag

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        #是否打印val_loss更新信息
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.savepath + 'checkpoint.pt')
        self.val_loss_min = val_loss