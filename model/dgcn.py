import torch
import torch.nn as nn
import numpy as np
from utils.adj_process import calculate_scaled_laplacian,calculate_random_walk_matrix
from torch.nn import functional as F

#参考DCRNN中的扩散图卷积模块
class DGCN(nn.Module):
    def __init__(self,adj_mx,input_dim,hidden_dim,num_nodes,max_diffusion_step,device,filter_type="dual_random_walk",nonlinearity='tanh'):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self.device = device

        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))

        for support in supports:
            self._supports.append(self._build_sparse_matrix(support, self.device))

        self._num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Ks
        input_size = input_dim + hidden_dim
        shape = (input_size*self._num_matrices,self._hidden_dim)
        self.weights = torch.nn.Parameter(torch.empty(*shape))
        self.biases = torch.nn.Parameter(torch.empty(self._hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.biases,0)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self,inputs,state_t):
        # 对X(t)和H(t-1)做图卷积，并加偏置bias
        # inputs and state_t(batch_size, num_nodes, input_dim/state_dim)
        batch_size,_,_=inputs.shape
        inputs_and_state = torch.cat([inputs,state_t],dim=2)
        inputs_size = inputs_and_state.shape[2] #=total_arg_size

        x = inputs_and_state
        #T0=I x0=T0*x
        x0 = x.permute(1,2,0)#[num_nodes,total_arg_size,batch_size]
        x0 = x0.reshape(self._num_nodes,-1)#[num_nodes,total_arg_size*batch_size]
        x = x0.unsqueeze(0)#[1,num_nodes,total_arg_size*batch_size]

        # 3阶[T0,T1,T2]Chebyshev多项式近似g(theta)
        # 把图卷积公式中的~L替换成了随机游走拉普拉斯D^(-1)*W
        # 第一类切比雪夫多项式
        """
        k-order Chebyshev polynomials : T0(L)~Tk(L)
        T0(L)=I/1 T1(L)=L Tk(L)=2LTk-1(L)-Tk-2(L)
        """
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                # T1=L x1=T1*x=L*x
                x1 = torch.sparse.mm(support,x0)# supports: n*n; x0: n*(total_arg_size * batch_size)
                x = self._concat(x,x1)
                for k in range(2,self._max_diffusion_step+1):
                    # T2=2LT1-T0=2L^2-1 x2=T2*x=2L^2x-x=2L*x1-x0...
                    # T3=2LT2-T1=2L(2L^2-1)-L x3=2L*x2-x1...
                    x2 = 2*torch.sparse.mm(support,x1) - x0
                    x = self._concat(x,x2)
                    x1,x0 = x2,x1 #循环
        # x.shape (Ks, num_nodes, total_arg_size * batch_size)
        # Ks = len(supports) * self._max_diffusion_step + 1
        x = x.reshape(self._num_matrices,self._num_nodes,inputs_size,batch_size)
        x = x.permute(3,1,2,0)# [batch_size, num_nodes, input_size, num_matrices]

        x = x.reshape(batch_size*self._num_nodes,inputs_size*self._num_matrices)
        x = torch.matmul(x,self.weights)
        x += self.biases#[batch_size,num_nodes,hidden_dim]
        x = self._activation(x)
        x = x.reshape(batch_size,self._num_nodes,self._hidden_dim)
        return x