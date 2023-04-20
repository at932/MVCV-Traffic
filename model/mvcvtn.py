import torch
import torch.nn as nn
from model.dgcn import DGCN
from model.gat import GAT

#MVCV-Traffic Network
class MVCVTNCell(nn.Module):
    def __init__(self,adj_mx,input_dim,hidden_dim,feat_dim,num_nodes,max_diffusion_step,num_heads,device):
        super(MVCVTNCell,self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._feat_dim = feat_dim
        self._num_nodes = num_nodes
        self.device = device
        self.alpha = 0.2
        self.residual = False

        self.gru_cell = nn.GRUCell(input_size=self._input_dim,hidden_size=self._hidden_dim)
        self.dgcn = DGCN(adj_mx=adj_mx,input_dim=self._input_dim,hidden_dim=self._hidden_dim,num_nodes=self._num_nodes,max_diffusion_step=max_diffusion_step,device=self.device)
        self.gat = GAT(adj_mx=adj_mx,input_dim=self._input_dim,hidden_dim=self._hidden_dim,feat_dim=self._feat_dim,num_nodes=self._num_nodes,num_heads=num_heads,max_diffusion_step=max_diffusion_step,alpha=self.alpha,residual=self.residual)

        #"""
        self.h_num = 3
        self.h_t_weights = nn.Parameter(torch.FloatTensor(self.h_num))
        self.softmax = nn.Softmax(dim=0)
        self.reset_parameters()
        #"""
        #self.gpu_tracker = MemTracker()
    def reset_parameters(self):
        nn.init.constant_(self.h_t_weights,1/self.h_num)

    #时间模块+空间模块+环境模块
    def forward(self, inputs, envs_feat, state_t, state_s, state_e):
        # [batch_size,num_nodes,input_dim/hidden_dim]
        batch_size, _, _ = inputs.shape
        w = self.softmax(self.h_t_weights)
        new_state_t = w[0] * state_t + w[1] * state_s + w[2]*state_e
        output_t = self.gru_cell(inputs.reshape(-1, self._input_dim), new_state_t.reshape(-1, self._hidden_dim))
        output_t = output_t.reshape(batch_size, self._num_nodes, self._hidden_dim)
        output_s = self.dgcn(inputs, output_t)
        #self.gpu_tracker.track()
        output_e = self.gat(inputs, envs_feat, output_t)
        #self.gpu_tracker.track()
        return output_t, output_s, output_e

#仅单向
class MVCVTN1(nn.Module):
    def __init__(self,adj_mx,input_dim,hidden_dim,feat_dim,seq_len,num_nodes,max_diffusion_step,num_heads,device):
        super(MVCVTN1,self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._feat_dim = feat_dim
        self._seq_len = seq_len
        self._num_nodes = num_nodes
        self.device = device

        self.mvcvtn_cell = MVCVTNCell(adj_mx=adj_mx,input_dim=self._input_dim,hidden_dim=self._hidden_dim,feat_dim=self._feat_dim,num_nodes=num_nodes,max_diffusion_step=max_diffusion_step,num_heads=num_heads,device=self.device)

    #时间模块+空间模块+环境模块
    def forward(self, inputs, envs_feat):
        batch_size, _, _ = inputs.shape
        state_t = torch.zeros((batch_size, self._num_nodes, self._hidden_dim), device=self.device)
        state_s = torch.zeros((batch_size, self._num_nodes, self._hidden_dim), device=self.device)
        state_e = torch.zeros((batch_size, self._num_nodes, self._hidden_dim), device=self.device)
        for i in range(self._seq_len + 1):  # 包括插值时刻
            # for i in range(self._seq_len): #不包括插值时刻,仅预测
            xi = inputs[:, :, i].unsqueeze(2)  # [batch_size,num_nodes,input_dim]
            output_t, output_s, output_e = self.mvcvtn_cell(xi, envs_feat, state_t, state_s,state_e)
            state_t = output_t
            state_s = output_s
            state_e = output_e
        return state_t, state_s, state_e

#双向
class MVCVTN2(nn.Module):
    def __init__(self,adj_mx,input_dim,hidden_dim,output_dim,feat_dim,seq_len,num_nodes,max_diffusion_step,num_heads,device):
        super(MVCVTN2,self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._feat_dim =feat_dim
        self._seq_len = seq_len #使用插值时刻前/后多少小时的时序
        self._num_nodes = num_nodes

        self.forwardMVCVTN = MVCVTN1(adj_mx=adj_mx,input_dim=self._input_dim,hidden_dim=self._hidden_dim,feat_dim=self._feat_dim,seq_len=self._seq_len,num_nodes=self._num_nodes,max_diffusion_step=max_diffusion_step,num_heads=num_heads,device=device)
        self.backwardMVCVTN = MVCVTN1(adj_mx=adj_mx,input_dim=self._input_dim,hidden_dim=self._hidden_dim,feat_dim=self._feat_dim,seq_len=self._seq_len,num_nodes=self._num_nodes,max_diffusion_step=max_diffusion_step,num_heads=num_heads,device=device)

        self.ouput_layer_t = nn.Linear(in_features=hidden_dim*2,out_features=output_dim)
        self.ouput_layer_s = nn.Linear(in_features=hidden_dim*2, out_features=output_dim)
        self.ouput_layer_e = nn.Linear(in_features=hidden_dim*2,out_features=output_dim)
        #self.gpu_tracker = MemTracker()

        # """
        #结果自适应加权
        self.y_num = 3
        self.y_weights = nn.Parameter(torch.FloatTensor(self.y_num))
        self.softmax = nn.Softmax(dim=0)
        self.reset_parameters()
        # """
    def reset_parameters(self):
        nn.init.constant_(self.y_weights,1/self.y_num)

    # 时间模块+空间模块+环境模块
    def forward(self, inputs, envs_feat):
        # self.gpu_tracker.track()
        # inputs [batch_size,num_nodes,seq_len=[t-T, ..., t, ..., t+T]共2T+1]
        inputs_fwd = inputs[:, :, :self._seq_len + 1]  # [t-T, t-T+1, ..., t]
        h_t_fwd, h_s_fwd, h_e_fwd = self.forwardMVCVTN(inputs_fwd, envs_feat)
        # self.gpu_tracker.track()

        inputs_bwd = inputs[:, :, self._seq_len:].flip(2)  # [t+T, t+T-1, ..., t]
        h_t_bwd, h_s_bwd, h_e_bwd = self.backwardMVCVTN(inputs_bwd, envs_feat)

        y_t = self.ouput_layer_t(torch.cat([h_t_fwd, h_t_bwd], dim=2))
        y_s = self.ouput_layer_s(torch.cat([h_s_fwd, h_s_bwd], dim=2))
        y_e = self.ouput_layer_e(torch.cat([h_e_fwd, h_e_bwd], dim=2))

        w = self.softmax(self.y_weights)
        y = w[0] * y_t + w[1] * y_s + w[2] * y_e

        y = y.squeeze(2)
        # self.gpu_tracker.track()
        y_t = y_t.squeeze(2)
        y_s = y_s.squeeze(2)
        y_e = y_e.squeeze(2)
        return y,y_t,y_s,y_e



