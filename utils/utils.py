import numpy as np
import torch

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def trans_normalization(data,max_val):
    """
    根据标准化保存的参数[最大值和最小值]逆标准化处理
    :param data:
    :param data_max:
    :param data_min:
    :return:
    """
    data2 = data * max_val
    return data2

def get_original_missing_mask(pred,label):
    """
    将原来就缺失的数据掩膜掉（数值小于1）
    :param pred:
    :param label:
    :return:
    """
    batch_num, road_num = pred.shape
    pred2 = pred.reshape(batch_num * road_num)
    label2 = label.reshape(batch_num * road_num)
    #mask = torch.nonzero(label2)
    mask = torch.where(label2 >= 1)
    pred2 = pred2[mask]
    label2 = label2[mask]
    return pred2,label2


def get_original_missing_mask_np(pred,label):
    """
    将原来就缺失的数据掩膜掉（数值小于1）
    :param pred:
    :param label:
    :return:
    """
    batch_num, road_num = pred.shape
    pred2 = pred.reshape(batch_num * road_num)
    label2 = label.reshape(batch_num * road_num)
    #mask = torch.nonzero(label2)
    mask = np.where(label2 >= 1)
    pred2 = pred2[mask]
    label2 = label2[mask]
    return pred2,label2

def get_data_by_mask(pred,label,mask):
    """
    获取缺失掩膜下的缺失数据，同时把原始数据中就缺失的路段掩膜掉
    :param data:
    :param mask:
    :return:
    """
    #data2 = torch.multiply(data,mask)

    # mask [batch_size,road_num]

    batch_num, road_num = mask.shape
    pred2 = pred.reshape(batch_num * road_num)
    label2 = label.reshape(batch_num * road_num)
    mask2 = mask.reshape(batch_num * road_num)

    #获取缺失路段的数据
    missing_index = torch.nonzero(mask2)
    pred2 = pred2[missing_index]
    label2 = label2[missing_index]

    #把原始数据中就缺失的路段掩膜掉,即只保留有效数据（数值大于等于1）
    data_index = torch.where(label2 >= 1)
    pred2 = pred2[data_index]
    label2 = label2[data_index]

    return pred2,label2




def get_data_by_mask_np(pred,label,mask):
    """
    获取缺失掩膜下的缺失数据，同时把原始数据中就缺失的路段掩膜掉
    :param data:
    :param mask:
    :return:
    """
    #data2 = torch.multiply(data,mask)

    # mask [batch_size,road_num]
    batch_num, road_num = mask.shape
    pred2 = pred.reshape(batch_num * road_num)
    label2 = label.reshape(batch_num * road_num)
    mask2 = mask.reshape(batch_num * road_num)

    #获取缺失路段的数据
    missing_index = np.nonzero(mask2)
    pred2 = pred2[missing_index]
    label2 = label2[missing_index]

    #把原始数据中就缺失的路段掩膜掉,即只保留有效数据（数值大于等于1）
    data_index = np.where(label2 >= 1)
    pred2 = pred2[data_index]
    label2 = label2[data_index]

    return pred2,label2