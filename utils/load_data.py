import numpy as np
import pandas as pd
import torch

def load_adj_data(path,dtype = np.float32):
    #path是所有data所处的文件夹，进到函数后加文件名
    adj_path = path + 'adj.csv'
    adj_df = pd.read_csv(adj_path,header=None)
    adj_array = np.array(adj_df,dtype=dtype)
    #[num_nodes, num_nodes]
    return adj_array

def load_speed_data(path,dtype = np.float32):
    speed_path = path + "speed.csv"
    speed_df = pd.read_csv(speed_path,header=None)
    speed_array = np.array(speed_df,dtype=dtype)
    #[num_noeds,time_len]
    return speed_array

def load_poi_feat_data(path):
    poi_feat_path = path + "poi_feat.csv"
    poi_feat_df = pd.read_csv(poi_feat_path)
    poi_feat_array = np.array(poi_feat_df)
    # [num_nodes,poi_feat_dim=14] #sz
    return poi_feat_array

def load_structure_feat_data(path):
    structure_feat_path = path + "structure_feat.csv"
    structure_feat_df = pd.read_csv(structure_feat_path)
    structure_feat_array = np.array(structure_feat_df)
    # [num_nodes,structure_feat_dim=6]
    return structure_feat_array

def load_missing_mask(path,miss_type,miss_rate,index_id,dtype = np.int16):
    missing_mask_path = path + "mask/missing_mask_{}{}_idx{}.csv".format(miss_type,miss_rate,index_id)
    missing_mask_df = pd.read_csv(missing_mask_path,header=None)
    missing_mask_array = np.array(missing_mask_df,dtype=dtype)
    #[num_nodes,time_len]
    return missing_mask_array

def drop_zero_data(speed_data,seq_len,count_threshold):
    """
    #把某些时刻原始缺失率很大的样本排除掉
    :param speed_data:
    :param seq_len: 输入序列长度
    :param count_threshold:阈值，如果在某一时刻原始缺失路段的数量超过这个阈值，则排除掉该时刻的样本
    :return:
    """
    df = pd.DataFrame(speed_data)
    drop_record = []
    #seq_len = 6  # 用前后1小时插值当前时刻
    zero_count = (df == 0).sum(axis=0)
    for i, count in enumerate(zero_count):
        if count > count_threshold:
            for j in range(0, seq_len + 1):
                drop_record.append(i - j)
                drop_record.append(i + j)
    drop_record = set(drop_record)  # 后面在这个drop_record里面的时刻标号就不参与训练样本生成了
    return drop_record

def generate_dataset(
        speed_data,missing_mask,envs_data,
        seq_len,count_threshold,split_ratio=[0.6,0.2,0.2],normalize=True
):
    """
    :param speed_data:
    :param missing_mask:
    :param poi_feat_data:
    :param stru_feat_data:
    :param seq_len:
    :param count_threshold
    :param split_ratio: 训练集、验证机和测试集划分比例
    :param normalize:数据是否标准化
    :return:
    """
    drop_record = drop_zero_data(speed_data,seq_len,count_threshold)
    time_len = speed_data.shape[1]-len(drop_record)-seq_len*2
    train_size = int(time_len*split_ratio[0])
    val_size = int(time_len*split_ratio[1])
    test_size = time_len-train_size-val_size

    if normalize:
        max_val = np.max(speed_data)
        speed_data = speed_data / max_val
    all_input, all_label, all_mask,all_feat = [],[],[],[]
    #time_slot_list = []
    for i in range(seq_len,speed_data.shape[1]-seq_len):
        #在drop_record里包含的时刻都是涉及到几乎所有路段都是缺失的
        if i in drop_record:
            continue
        #"""
        #time_slot_list.append(i)
        speed_i = np.copy(speed_data[:, i])
        missing_mask_i = np.copy(missing_mask[:,i])
        observed_mask_i = np.where(missing_mask_i==0,1,0)
        observed_speed_i = np.multiply(speed_i,observed_mask_i)

        speed_range = np.copy(speed_data[:,i-seq_len:i+seq_len+1])
        speed_range[:,seq_len] = observed_speed_i
        #"""


        all_input.append(speed_range)
        all_label.append(speed_i)
        all_mask.append(missing_mask_i)
        all_feat.append(envs_data)

    train_input = all_input[:train_size]
    train_label = all_label[:train_size]
    train_mask = all_mask[:train_size]
    train_feat = all_feat[:train_size]

    val_input = all_input[train_size:train_size+val_size]
    val_label = all_label[train_size:train_size+val_size]
    val_mask = all_mask[train_size:train_size+val_size]
    val_feat = all_feat[train_size:train_size+val_size]

    test_input = all_input[train_size+val_size:time_len]
    test_label = all_label[train_size+val_size:time_len]
    test_mask = all_mask[train_size+val_size:time_len]
    test_feat = all_feat[train_size+val_size:time_len]


    return np.array(train_input),np.array(train_label),np.array(train_mask),np.array(train_feat),np.array(val_input),np.array(val_label),np.array(val_mask),np.array(val_feat),np.array(test_input),np.array(test_label),np.array(test_mask),np.array(test_feat)

def generate_torch_dataset(
        speed_data,missing_mask,envs_data,
        seq_len,count_threshold,split_ratio=[0.6,0.2,0.2],normalize=True
):
    train_input, train_label, train_mask, train_feat, val_input, val_label, val_mask, val_feat, test_input, test_label, test_mask, test_feat = generate_dataset(
        speed_data, missing_mask, envs_data, seq_len,count_threshold)

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_input),torch.FloatTensor(train_label),torch.FloatTensor(train_mask),torch.FloatTensor(train_feat)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_input), torch.FloatTensor(val_label), torch.FloatTensor(val_mask),torch.FloatTensor(val_feat)
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_input), torch.FloatTensor(test_label), torch.FloatTensor(test_mask), torch.FloatTensor(test_feat)
    )
    return train_dataset, val_dataset,test_dataset

def envs_feat_process(poi_feat_data, structure_feat_data):
    """
    处理环境因子，拼接整合成一个向量
    :param poi_feat_data:
    :param structure_feat_data:
    :return:
    """
    #原来直接合并在一起
    #feat = np.concatenate([poi_feat_data, structure_feat_data], axis=1)  # (23,) dim=23
    #poi_feat [餐饮服务,公司企业,交通设施服务,科教文化服务,汽车服务,生活服务,医疗保健服务,体育休闲服务,政府机构及社会团体,住宿服务,商务住宅,购物服务,金融保险服务,风景名胜]
    #structure_feat [length,centrality,betweenness,closeness,curve,level]
    #poi特征经过tf-idf加权处理，因此这里仅对structure特征进行处理
    #先写成固定的了
    new_structure_feat_data = np.zeros((structure_feat_data.shape[0],9))
    for i in range(structure_feat_data.shape[1]-1):
        new_structure_feat_data[:,i] = (structure_feat_data[:,i] - np.min(structure_feat_data[:,i])) / (np.max(structure_feat_data[:,i]) - np.min(structure_feat_data[:,i]))
    i = structure_feat_data.shape[1]-1
    #等级编码为one-hot向量
    for j in range(structure_feat_data.shape[0]):
        new_structure_feat_data[j,i+structure_feat_data[j,i].astype(int)-1] = 1

    feat = np.concatenate([poi_feat_data, new_structure_feat_data], axis=1)  # (20,) dim=23
    return feat


#包括train、val和test
def load_data(path, index_id,miss_type,miss_rate,dataset_name,seq_len):
    path = path + "{}_data/".format(dataset_name)
    count_threshold = 200

    speed_data = load_speed_data(path)
    max_val = np.max(speed_data)
    missing_mask = load_missing_mask(path,miss_type,miss_rate,index_id)

    if dataset_name == 'sz':
        poi_feat_data = load_poi_feat_data(path)
        structure_feat_data = load_structure_feat_data(path)
        envs_data = envs_feat_process(poi_feat_data,structure_feat_data)
    elif dataset_name == 'metr_la':
        poi_feat_data = load_poi_feat_data(path)
        envs_data = poi_feat_data


    adj = load_adj_data(path)

    train_dataset, val_dataset,test_dataset = generate_torch_dataset(speed_data, missing_mask, envs_data, seq_len,count_threshold)

    return train_dataset, val_dataset,test_dataset, adj, max_val


