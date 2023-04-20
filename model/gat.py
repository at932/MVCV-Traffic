import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.adj_process import calculate_random_walk_matrix

def adj_to_bias(adj_mx,max_diffusion_step):
    """
    选择归一化约束的节点范围(邻居节点、非邻居节点）
    #默认dgcn的filter type是dual_random_walk
    :param adj/A:[num_nodes,num_nodes]
    :param max_diffusion_step/k:扩散图卷积的最大扩散步数
    :return:
    """
    filter_type = "dual_random_walk"
    supports = [] #dense版本
    if filter_type == "dual_random_walk":
        supports.append((calculate_random_walk_matrix(adj_mx).T).toarray())
        supports.append((calculate_random_walk_matrix(adj_mx.T).T).toarray())

    supports_k_order = []
    for support in supports:
        p = support
        for k in range(2,max_diffusion_step+1):
            p = np.matmul(p,support)
        supports_k_order.append(p)

    mask = np.zeros((supports_k_order[0].shape[0],supports_k_order[0].shape[1]))
    for sup in supports_k_order:
        mask = mask + sup

    #mask =(D_out * A)^k + (D_in * A)^k
    mask = np.where(mask>0,1,0)
    #掩膜掉扩散图卷积里的k阶邻域
    return -1e9 * mask
    #return -1e9 * (1.0 - mask)


class GraphAttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim,num_nodes,alpha,concat=True,residual=False):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._num_nodes = num_nodes
        self.concat = concat
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.residual = residual

        self.W = nn.Parameter(torch.empty((self._input_dim,self._output_dim)))
        self.a1 = nn.Parameter(torch.empty(self._output_dim,1))
        self.a2 = nn.Parameter(torch.empty(self._output_dim,1))
        """
        if self.residual:
            if self._input_dim != self._output_dim:
                self.res_fc = nn.Linear(self._input_dim,self._output_dim,bias=False)
        """

        self.softmax = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a1, gain=1.414)
        nn.init.xavier_uniform_(self.a2, gain=1.414)

    def forward(self,h,bias_mat):
        batch_size,_,_=h.shape
        Wh = torch.matmul(h,self.W) #转换维度至output_dim

        e = self._prepare_attentional_mechanism_input(Wh) + bias_mat #mask掉邻接部分
        #print(e[0,1,2])
        attention = self.softmax(e) #[B,N,N]
        #print(attention[0, 50, :])
        #if not self.concat:
        #    print(torch.max(attention[0,50,:]))

        #h_prime = torch.matmul(attention,Wh)#和下面那一行都是一样的,在batch里广播
        h_prime = torch.bmm(attention,Wh) #以行为单位，一行代表1个结点

        """
        # residual connection??
        if self.residual:
            if h.shape[-1] != h_prime.shape[-1]:
                h_prime = h_prime + self.res_fc(h.reshape(-1,self._input_dim)).reshape(batch_size,self._num_nodes,self._output_dim)
            else:
                h_prime = h_prime + h
        """
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh,self.a1) #[B,N,1]
        Wh2 = torch.matmul(Wh,self.a2) #[B,N,1]
        # broadcast add
        e = Wh1 + Wh2.permute(0,2,1)#[B,N,1]+[B,1,N],广播/broadcast
        return self.leakyrelu(e)

class GAT(nn.Module):
    def __init__(self,adj_mx,input_dim,hidden_dim,feat_dim,num_nodes,num_heads,max_diffusion_step,alpha,residual=False):
        super().__init__()
        self._input_dim = input_dim #输入交通流特征维度
        self._hidden_dim = hidden_dim #隐藏单元维度
        self._feat_dim = feat_dim #环境特征维度
        self._num_nodes = num_nodes #道路数量
        self._num_heads = num_heads #多头注意力的个数
        self._alpha = alpha # leaky_relu激活函数的参数
        self.register_buffer('bias_mat',torch.FloatTensor(adj_to_bias(adj_mx,max_diffusion_step)))
        #self.bias_mat = torch.FloatTensor(adj_yo_bias(adj_mx)) #邻接矩阵变成偏置矩阵
        self._k_att_input_dim = input_dim+feat_dim+hidden_dim

        #self.act = nn.Tanh()
        self.act = nn.ELU()

        self.attentions = [GraphAttentionLayer(input_dim=self._k_att_input_dim,output_dim=self._hidden_dim,num_nodes=self._num_nodes,alpha=self._alpha,concat=True,residual=residual)
                           for _ in range(self._num_heads)]

        for i,attention in enumerate(self.attentions):
            self.add_module('attetion_{}'.format(i),attention)

        #输出层
        self.out_att = GraphAttentionLayer(input_dim=self._hidden_dim*self._num_heads,output_dim=self._hidden_dim,num_nodes=self._num_nodes,alpha=self._alpha,concat=False,residual=residual)

    def forward(self,inputs,envs_feat,state_t):
        # inputs [batch_size,num_nodes,1]
        # envs_feat [batch_size,num_nodes,feat_dim=23]
        # state_t [batch_size,num_nodes,hidden_dim]
        # adj邻接矩阵→偏置矩阵[num_nodes,num_nodes]
        batch_size,_,_ = inputs.shape
        h = torch.cat([inputs,envs_feat,state_t],dim=2)
        h_1 = torch.cat([self.act(att(h,self.bias_mat)) for att in self.attentions],dim=2)
        out = self.act(self.out_att(h_1,self.bias_mat))

        return out

