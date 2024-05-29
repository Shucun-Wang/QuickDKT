import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding, Linear, Dropout


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DKVMN(nn.Module):
    def __init__(self, m_size, n_q, k_dim, v_m_dim, v_dim, f_dim, dropout=0.5):
        super(DKVMN, self).__init__()

        self.m_size = m_size
        self.n_q = n_q
        self.k_dim = k_dim
        self.v_m_dim = v_m_dim
        self.v_dim = v_dim
        self.f_dim = f_dim

        self.dropout = Dropout(dropout)

        # read layer
        self.k_e_layer = Embedding(self.n_q + 1, self.k_dim)
        self.f_layer = Linear(self.k_dim + self.v_m_dim, self.f_dim)
        self.p_layer = Linear(self.f_dim, 1) # output layer

        nn.init.kaiming_normal_(self.k_e_layer.weight)
        nn.init.kaiming_normal_(self.f_layer.weight)
        nn.init.kaiming_normal_(self.p_layer.weight)

        nn.init.constant_(self.f_layer.bias, 0)
        nn.init.constant_(self.p_layer.bias, 0)

        # write layer
        self.v_e_layer = Embedding(2 * self.n_q + 1, self.v_dim)
        self.e_layer = Linear(self.v_dim, self.v_m_dim)
        self.a_layer = Linear(self.v_dim, self.v_m_dim)

        nn.init.kaiming_normal_(self.v_e_layer.weight)
        nn.init.kaiming_normal_(self.e_layer.weight)
        nn.init.kaiming_normal_(self.a_layer.weight)

        # key, value memory and initialize
        self.Mk = Parameter(torch.randn(self.m_size, self.k_dim))
        self.Mv0 = Parameter(torch.randn(self.m_size, self.v_m_dim))

        nn.init.kaiming_normal_(self.Mk)
        nn.init.kaiming_normal_(self.Mv0)

    def forward(self, q, qa):
        
        batch_size = q.shape[0]
        seqlen = q.shape[1]

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1).detach()
        # (bs, 20, 200)
        
        s_q = torch.chunk(q, seqlen, 1) 
        s_qa = torch.chunk(qa, seqlen, 1)
        # (bs, 1, em_dim)

        output_p = torch.Tensor([]).to(device)

        for i in range(seqlen):
            q = s_q[i].squeeze(1)
            qa = s_qa[i].squeeze(1)
            # (bs, em_dim)

            k = self.k_e_layer(q)
            v = self.v_e_layer(qa)
            
            e_input = v
            a_input = v

            # correlation weight 计算该题与各个知识概念的关系
            w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)
            # (64, 50) * (20, 50).T = (64, 20)

            # read process

            r = (w.unsqueeze(-1) * Mvt).sum(1)
            f = torch.tanh(self.f_layer(torch.cat([r, k], dim=-1)))
            p = self.p_layer(f)
            output_p = torch.cat([output_p, p], dim=1)

            # write process
            e = torch.sigmoid(self.e_layer(e_input))
            a = torch.tanh(self.a_layer(a_input))

            e_reshape = e.view(-1, 1, self.v_m_dim)
            a_reshape = a.view(-1, 1, self.v_m_dim)
            w_reshape = w.view(-1, self.m_size, 1)

            Mvt = Mvt * (1 - e_reshape * w_reshape) + a_reshape * w_reshape

        return output_p



