# -*- coding:utf-8 -*-
import time
import argparse
import torchmetrics
import torch
import torch.nn as nn
import numpy as np
from utils.load_data import load_data
from utils.earlystopping import EarlyStopping
from utils.utils import trans_normalization,DeviceDataLoader,get_original_missing_mask,get_data_by_mask
from model.mvcvtn import MVCVTN2

def train(model,dataloader, loss_fun, optimizer, max_val):
    #print('train')
    train_loss = 0
    num_batches = len(dataloader)
    model.train()
    #print(num_batches)
    for batch_idx, data in enumerate(dataloader):
        #print("batch {}/{}".format(batch_idx,num_batches))
        inputs, labels, masks, feats = data

        pred,y_t,y_s,y_e = model(inputs,feats)
        pred = trans_normalization(pred, max_val)
        labels = trans_normalization(labels, max_val)
        y_t = trans_normalization(y_t,max_val)
        y_s = trans_normalization(y_s,max_val)
        y_e = trans_normalization(y_e,max_val)

        # 原来为0（小于等于1的）掩膜掉，避免其影响mape
        pred2, labels2 = get_original_missing_mask(pred, labels)
        y_t2, _ = get_original_missing_mask(y_t, labels)
        y_s2, _ = get_original_missing_mask(y_s, labels)
        y_e2, _ = get_original_missing_mask(y_e, labels)

        loss = loss_fun(pred2,labels2) + loss_fun(y_t2,labels2) + loss_fun(y_s2,labels2) + loss_fun(y_e2,labels2)

        #compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
    train_loss /= num_batches
    return train_loss

def test(model,dataloader, loss_fun, max_val):
    #print('test')
    test_loss, test_mape, test_mae, test_rmse = 0, 0, 0, 0
    num_batches = len(dataloader)
    model.eval()  # 开启测试模型
    #print(num_batches)
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            #print("batch {}/{}".format(batch_idx,num_batches))
            inputs, labels, masks, feats = data

            pred, y_t, y_s, y_e = model(inputs, feats)
            pred = trans_normalization(pred, max_val)
            labels = trans_normalization(labels, max_val)
            y_t = trans_normalization(y_t, max_val)
            y_s = trans_normalization(y_s, max_val)
            y_e = trans_normalization(y_e, max_val)

            # 原来为0（小于等于1的）掩膜掉，避免其影响mape
            pred2, labels2 = get_original_missing_mask(pred, labels)
            y_t2, _ = get_original_missing_mask(y_t, labels)
            y_s2, _ = get_original_missing_mask(y_s, labels)
            y_e2, _ = get_original_missing_mask(y_e, labels)

            loss = loss_fun(pred2, labels2) + loss_fun(y_t2, labels2) + loss_fun(y_s2, labels2) + loss_fun(y_e2,labels2)

            mape = torchmetrics.functional.mean_absolute_percentage_error(pred2, labels2)
            mae = torchmetrics.functional.mean_absolute_error(pred2, labels2)
            rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(pred2, labels2))

            test_loss += loss.item()
            test_mape += mape.item()
            test_mae += mae.item()
            test_rmse += rmse.item()

    test_loss /= num_batches
    test_mape /= num_batches
    test_mae /= num_batches
    test_rmse /= num_batches

    return test_loss,test_mape,test_mae,test_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='')  # 迭代次数
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')  # 学习率
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='')  # 衰减率，优化器参数#5e-5 #1e-4 #1e-5
    parser.add_argument('--batch_size', type=int, default=100, help='')  # 批量数

    opt = parser.parse_args()
    epochs = opt.epochs
    learning_rate = opt.learning_rate  # 学习率
    weight_decay = opt.weight_decay  # 衰减率
    batch_size = opt.batch_size

    #early stopping的参数
    patience = 10
    verbose = False
    delta = 0

    input_dim = 1
    output_dim = 1

    dataset_name = 'metr_la' #sz metra_la

    if dataset_name == 'metr_la':
        seq_len = 12
        hidden_dim = 64
        feat_dim = 9
        max_diffusion_step = 2
        nheads = 3

    miss_type = 'rm' #'rm' 'bm' 'mm' #缺失类型
    miss_rate = 20 #20 40 60 #缺失率
    index_id = 1 #表示某种缺失模式[缺失类型，缺失率]的第i个实例，目前都只有1个实例
    print(miss_type, miss_rate)
    path = "./"

    model_savepath = path + "mvcvtn_{}{}.pt".format(miss_type, miss_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, _, adj, max_val = load_data(path=path, index_id=index_id, miss_type=miss_type,miss_rate=miss_rate, dataset_name=dataset_name,seq_len=seq_len)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DeviceDataLoader(train_dataloader, device)
    val_dataloader = DeviceDataLoader(val_dataloader, device)


    model = MVCVTN2(adj_mx=adj, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, feat_dim=feat_dim,
                  seq_len=seq_len, num_nodes=adj.shape[0], max_diffusion_step=max_diffusion_step, num_heads=nheads,
                  device=device)
    model.to(device)

    loss_fun = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    early_stopping = EarlyStopping(model_savepath, patience, verbose, delta)

    train_loss_list, test_loss_list, test_mape_list, test_mae_list, test_rmse_list = list(), list(), list(), list(), list()

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_dataloader, loss_fun, optimizer, max_val)

        test_loss, test_mape, test_mae, test_rmse = test(model, val_dataloader, loss_fun, max_val)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_mape_list.append(test_mape)
        test_mae_list.append(test_mae)
        test_rmse_list.append(test_rmse)

        print('Epoch: %d | train_loss: %.5f  | test_loss: %.5f | test_mape: %.5f |test_mae: %.5f | test_rmse: %.5f'
              % (epoch, train_loss, test_loss, test_mape, test_mae, test_rmse))

        save_flag = early_stopping(test_loss, model)
        if save_flag:
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': test_loss,
                'mape': test_mape,
                'mae': test_mae,
                'rmse': test_rmse,
            }

            torch.save(state, model_savepath)
        end_time = time.time()
        print("epoch time:{}s".format(end_time - start_time))
        #if early_stopping.early_stop:
        #    break

    print("Done!")
    index = test_rmse_list.index(np.min(test_rmse_list))
    print('min_rmse:%.4f' % (test_rmse_list[index]),
          'min_mape:%.4f' % (test_mape_list[index]),
          'min_mae:%.4f' % (test_mae_list[index]))


