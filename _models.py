import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import geoopt.manifolds.poincare.math as pmath_geo
import math, itertools


class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, hidden_states, reverse=False):

        b, seq, embed = inputs.size()
        h = hidden_states[0]
        c = hidden_states[1]

        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        hidden_state_h = []
        hidden_state_c = []

        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))  # short term mem
            # discounted short term mem
            c_s2 = c_s1 * timestamps[:, s: s + 1].expand_as(c_s1)
            c_l = c - c_s1  # long term mem
            c_adj = c_l + c_s2  # adjusted = long + disc short term mem
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(o)
            hidden_state_c.append(c)
            hidden_state_h.append(h)

        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()

        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)

        return outputs, (h, c)


class SimpleAttn(torch.nn.Module):
    def __init__(self, in_shape, use_attention=True, maxlen=None):
        super(SimpleAttn, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.W1 = torch.nn.Linear(in_shape, in_shape)
            self.W2 = torch.nn.Linear(in_shape, in_shape)
            self.V = torch.nn.Linear(in_shape, 1)
        if maxlen is not None:
            self.arange = torch.arange(maxlen)

    def forward(self, full, last, lens=None, dim=1):
        """
        full : B*30*in_shape
        last : B*1*in_shape
        lens: B*1
        """
        if self.use_attention:
            score = self.V(F.tanh(self.W1(last) + self.W2(full)))

            if lens != None:
                mask = self.arange[None, :] < lens[:, None]
                score[~mask] = float("-inf")

            attention_weights = F.softmax(score, dim=dim)
            context_vector = attention_weights * full
            context_vector = torch.sum(context_vector, dim=dim)
            return context_vector
        else:
            if lens != None:
                mask = self.arange[None, :] < lens[:, None]
                mask = mask.type(torch.float).unsqueeze(-1).cuda()
                context_vector = full * mask
                context_vector = torch.mean(context_vector, dim=dim)
                return context_vector
            else:
                return torch.mean(full, dim=dim)


def one_rnn_transform(W, h, U, x, c):
    W_otimes_h = pmath_geo.mobius_matvec(W, h, c=c)
    U_otimes_x = pmath_geo.mobius_matvec(U, x, c=c)
    Wh_plus_Ux = pmath_geo.mobius_add(W_otimes_h, U_otimes_x, c=c)
    return Wh_plus_Ux


class TimeLSTMHyp(nn.Module):
    def __init__(
            self, input_size, hidden_size, device, cuda_flag=False, bidirectional=False
    ):
        super(TimeLSTMHyp, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.U_all = torch.nn.Parameter(torch.Tensor(hidden_size * 4, input_size))
        self.W_d = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bidirectional = bidirectional
        self.c = torch.tensor([1.0]).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.W_all, self.U_all, self.W_d]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def proj(self, x, c):
        norm = torch.clamp_min(
            x.norm(dim=-1, keepdim=True, p=2), 1e-15)
        maxnorm = (1 - 1e-5) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
        
    def forward(self, inputs, timestamps, hidden_states, reverse=False):
        b, seq, embed = inputs.size()
        h = hidden_states[0]
        _c = hidden_states[1]
        if self.cuda_flag:
            h = h.cuda()
            _c = _c.cuda()
        outputs = []
        hidden_state_h = []
        hidden_state_c = []

        for s in range(seq):
            c_s1 = pmath_geo.expmap0(
                torch.tanh(
                    pmath_geo.logmap0(
                        pmath_geo.mobius_matvec(self.W_d, _c, c=self.c), c=self.c
                    )
                )
            )  # short term mem
            c_s2 = pmath_geo.mobius_pointwise_mul(
                c_s1, timestamps[:, s: s + 1].expand_as(c_s1), c=self.c
            )  # discounted short term mem
            c_l = pmath_geo.mobius_add(-c_s1, _c, c=self.c)  # long term mem
            c_adj = pmath_geo.mobius_add(c_l, c_s2, c=self.c)

            W_f, W_i, W_o, W_c_tmp = self.W_all.chunk(4, dim=1)
            U_f, U_i, U_o, U_c_tmp = self.U_all.chunk(4, dim=0)
            # print ('WF: ', W_f.shape)
            # print ('H: ', h.shape)
            # print ('UF: ', U_f.shape)
            # print ('X: ', inputs[:, s].shape)
            f = pmath_geo.logmap0(
                one_rnn_transform(W_f, h, U_f, inputs[:, s], self.c), c=self.c
            ).sigmoid()
            i = pmath_geo.logmap0(
                one_rnn_transform(W_i, h, U_i, inputs[:, s], self.c), c=self.c
            ).sigmoid()
            o = pmath_geo.logmap0(
                one_rnn_transform(W_o, h, U_o, inputs[:, s], self.c), c=self.c
            ).sigmoid()
            c_tmp = pmath_geo.logmap0(
                one_rnn_transform(W_c_tmp, h, U_c_tmp, inputs[:, s], self.c), c=self.c
            ).sigmoid()

            f_dot_c_adj = pmath_geo.mobius_pointwise_mul(f, c_adj, c=self.c)
            i_dot_c_tmp = pmath_geo.mobius_pointwise_mul(i, c_tmp, c=self.c)
            _c = pmath_geo.mobius_add(i_dot_c_tmp, f_dot_c_adj, c=self.c)

            h = pmath_geo.mobius_pointwise_mul(
                o, pmath_geo.expmap0(torch.tanh(_c), c=self.c), c=self.c
            )
            outputs.append(o)
            hidden_state_c.append(_c)
            hidden_state_h.append(h)

        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()
        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)

        return outputs, (h, _c)



class TimeLSTMHypV1(nn.Module):
    def __init__(
            self, input_size, hidden_size, device, cuda_flag=False, bidirectional=False
    ):
        super(TimeLSTMHypV1, self).__init__()
        print("Its me V1")
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.U_all = torch.nn.Parameter(torch.Tensor(hidden_size * 4, input_size))
        self.W_d = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bidirectional = bidirectional
        self.c = torch.tensor([1.0]).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.W_all, self.U_all, self.W_d]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def proj(self, x, c):
        norm = torch.clamp_min(
            x.norm(dim=-1, keepdim=True, p=2), 1e-15)
        maxnorm = (1 - 1e-5) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
        
    def forward(self, inputs, timestamps, hidden_states, reverse=False):
        b, seq, embed = inputs.size()
        h = hidden_states[0]
        _c = hidden_states[1]
        if self.cuda_flag:
            h = h.cuda()
            _c = _c.cuda()
        outputs = []
        hidden_state_h = []
        hidden_state_c = []

        for s in range(seq):
            c_s1 = self.proj(pmath_geo.expmap0(
                torch.tanh(
                    pmath_geo.logmap0(
                        pmath_geo.mobius_matvec(self.W_d, _c, c=self.c), c=self.c
                    )
                )
            ), c=self.c)  # short term mem
            c_s2 = self.proj(pmath_geo.mobius_pointwise_mul(
                c_s1, timestamps[:, s: s + 1].expand_as(c_s1), c=self.c
            ), c=self.c)  # discounted short term mem
            c_l = self.proj(pmath_geo.mobius_add(-c_s1, _c, c=self.c), c=self.c)  # long term mem
            c_adj = self.proj(pmath_geo.mobius_add(c_l, c_s2, c=self.c), c= self.c)

            W_f, W_i, W_o, W_c_tmp = self.W_all.chunk(4, dim=1)
            U_f, U_i, U_o, U_c_tmp = self.U_all.chunk(4, dim=0)
            # print ('WF: ', W_f.shape)
            # print ('H: ', h.shape)
            # print ('UF: ', U_f.shape)
            # print ('X: ', inputs[:, s].shape)
            f = pmath_geo.logmap0(
                one_rnn_transform(W_f, h, U_f, inputs[:, s], self.c), c=self.c
            ).sigmoid()
            i = pmath_geo.logmap0(
                one_rnn_transform(W_i, h, U_i, inputs[:, s], self.c), c=self.c
            ).sigmoid()
            o = pmath_geo.logmap0(
                one_rnn_transform(W_o, h, U_o, inputs[:, s], self.c), c=self.c
            ).sigmoid()
            c_tmp = pmath_geo.logmap0(
                one_rnn_transform(W_c_tmp, h, U_c_tmp, inputs[:, s], self.c), c=self.c
            ).sigmoid()

            f_dot_c_adj = self.proj(pmath_geo.mobius_pointwise_mul(f, c_adj, c=self.c), c=self.c)
            i_dot_c_tmp = self.proj(pmath_geo.mobius_pointwise_mul(i, c_tmp, c=self.c), c=self.c)
            _c = self.proj(pmath_geo.mobius_add(i_dot_c_tmp, f_dot_c_adj, c=self.c), c=self.c)

            h = self.proj(pmath_geo.mobius_pointwise_mul(
                o, pmath_geo.expmap0(torch.tanh(_c), c=self.c), c=self.c
            ), c=self.c)
            outputs.append(o)
            hidden_state_c.append(_c)
            hidden_state_h.append(h)

        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()
        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)

        return outputs, (h, _c)
