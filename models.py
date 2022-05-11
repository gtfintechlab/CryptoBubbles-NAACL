from _models import *
import torch
import torch.nn as nn
import geoopt.manifolds.poincare.math as pmath_geo
from mobius.mobius_gru import MobiusGRU


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(LSTMEncoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, batch_first=True)

    def forward(self, src):
        outputs, (hidden, cell) = self.rnn(src)

        return hidden.squeeze(0), cell.squeeze(0)


class LSTMEncoderAttn(nn.Module):
    def __init__(self, input_dim, hid_dim, maxlen=75):
        super(LSTMEncoderAttn, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, batch_first=True)
        self.attn = SimpleAttn(hid_dim, maxlen=maxlen, use_attention=True)

    def forward(self, src, len_feats):
        outputs, (hidden, cell) = self.rnn(src)
        hidden = hidden.permute(1, 0, 2)
        cell = cell.permute(1, 0, 2)
        hidden = self.attn(outputs, hidden, len_feats)
        cell = self.attn(outputs, cell, len_feats)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_dim, hid_dim, num_span_classes, out_dim=2, bs=16, num_days=20
    ):
        super(Decoder, self).__init__()

        self.hid_dim = hid_dim
        self.num_days = num_days
        self.input_dim = input_dim
        self.rnn_cell = nn.LSTMCell(input_dim, hid_dim)
        self.fc_in = nn.Linear(hid_dim, input_dim)
        self.fc_out = nn.Linear(hid_dim, out_dim)
        self.num_span_layer = nn.Linear(hid_dim, num_span_classes)
        self.LogSoftmax = nn.LogSoftmax(dim=-1)
        self.Softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, hx, cx):
        """
        hx: (batch_size, hiddem_dim)
        cx: (batch_size, hiddem_dim)
        """
        bs, hid_dim = hx.shape
        num_spans = self.Softmax(self.num_span_layer(hx))

        input = torch.zeros(size=(bs, self.input_dim)).cuda()
        outputs = []
        for i in range(self.num_days):
            hx, cx = self.rnn_cell(input, (hx, cx))
            input = self.relu(self.fc_in(hx))
            # print(hx.shape)
            outputs.append(self.Softmax(self.fc_out(hx)))

        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute(1, 0, 2)
        # outputs.shape = (batch_size, num_days, 2)
        return num_spans, outputs


class TimeLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, bs):
        """
        return_last -> to return only the last hidden/cell state
        """
        super(TimeLSTMEncoder, self).__init__()
        self.rnn = TimeLSTM(input_dim, hid_dim)
        self.bs = bs
        self.hid_dim = hid_dim
        self.device = torch.device("cuda:0")

    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.hid_dim)).to(self.device)
        c = Variable(torch.zeros(self.bs, self.hid_dim)).to(self.device)

        return (h, c)

    def forward(self, sentence_feats, time_feats):
        """
        sentence_feat = (B*75*N)
        time_feats = (B*75)
        """

        h_init, c_init = self.init_hidden()
        lstmout, (h_out, c_out) = self.rnn(
            sentence_feats, time_feats, (h_init, c_init))
        return h_out, c_out


class TimeLSTMAttnEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, bs, maxlen=75):
        super(TimeLSTMAttnEncoder, self).__init__()
        self.rnn = TimeLSTM(input_dim, hid_dim)
        self.attn = SimpleAttn(hid_dim, maxlen=maxlen, use_attention=True)
        self.bs = bs
        self.hid_dim = hid_dim
        self.device = torch.device("cuda:0")

    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.hid_dim)).to(self.device)
        c = Variable(torch.zeros(self.bs, self.hid_dim)).to(self.device)

        return (h, c)

    def forward(self, sentence_feats, time_feats, len_feats):
        """
        sentence_feat = (B*75*N)
        time_feats = (B*75)
        len_feats = (B)
        """

        h_init, c_init = self.init_hidden()

        lstmout, (h_out, c_out) = self.rnn(
            sentence_feats, time_feats, (h_init, c_init))
        # print(lstmout.shape)
        # print(h_out.shape)
        # print(len_feats.shape)

        h_out = self.attn(lstmout, h_out.unsqueeze(1), len_feats)
        c_out = self.attn(lstmout, c_out.unsqueeze(1), len_feats)

        return h_out, c_out


class TimeHypLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, bs, no_time=True):
        super(TimeHypLSTMEncoder, self).__init__()
        self.bs = bs
        self.no_time = no_time
        self.hid_dim = hid_dim
        self.device = torch.device("cuda:0")
        self.rnn = TimeLSTMHyp(input_dim, hid_dim, self.device, cuda_flag=True)
        self.c = torch.tensor([1.0]).to(self.device)
        self.tanh = nn.Tanh()

    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.hid_dim)).to(self.device)
        c = Variable(torch.zeros(self.bs, self.hid_dim)).to(self.device)

        return h, c

    def proj(self, x, c):
        norm = torch.clamp_min(
            x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def forward(self, sentence_feats, time_feats):
        """
        sentence_feat = (B*75*N)
        time_feats = (B*75)
        """
        # log -> tanh -> expmap -> encoder
        # sentence_feats = pmath_geo.logmap0(sentence_feats, c=self.c)
        # sentence_feats = self.tanh(sentence_feats)
        sentence_feats = pmath_geo.expmap0(sentence_feats, c=self.c)
        sentence_feats = pmath_geo.project(sentence_feats)
        # time_feats = pmath_geo.expmap0(time_feats, c=self.c)
        h_init, c_init = self.init_hidden()
        # h_init = pmath_geo.expmap0(h_init, c=self.c)
        # c_init = pmath_geo.expmap0(c_init, c=self.c)

        if self.no_time:
            time_feats = torch.ones_like(time_feats).cuda()

        _, (h_out, c_out) = self.rnn(
            sentence_feats, time_feats, (h_init, c_init))
        h_out = pmath_geo.logmap0(h_out, c=self.c)
        c_out = pmath_geo.logmap0(c_out, c=self.c)
        return h_out, c_out


class FullModel(nn.Module):
    def __init__(self, input_dim, hid_dim, bs, no_time=False):
        super(FullModel, self).__init__()
        self.bs = bs
        self.no_time = no_time
        self.hid_dim = hid_dim
        self.device = torch.device("cuda:0")
        self.rnn = TimeLSTMHypV1(
            input_dim, hid_dim, self.device, cuda_flag=True)
        self.c = torch.tensor([1.0]).to(self.device)
        self.tanh = nn.Tanh()
        self.decoder = Decoder(input_dim, hid_dim, 4,
                               out_dim=2, bs=self.bs, num_days=10)
        self.linear1 = nn.Linear(hid_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.hid_dim)).to(self.device)
        c = Variable(torch.zeros(self.bs, self.hid_dim)).to(self.device)

        return h, c

    def proj(self, x, c):
        norm = torch.clamp_min(
            x.norm(dim=-1, keepdim=True, p=2), 1e-15)
        maxnorm = (1 - 1e-5) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def forward(self, sentence_feats, time_feats):
        """
        sentence_feat = (B*75*N)
        time_feats = (B*75)
        """
        # log -> tanh -> expmap -> encoder
        # sentence_feats = pmath_geo.logmap0(sentence_feats, c=self.c)
        # sentence_feats = self.tanh(sentence_feats)
        sentence_feats = pmath_geo.expmap0(sentence_feats, c=self.c)
        sentence_feats = self.proj(sentence_feats, c=self.c)
        time_feats = pmath_geo.expmap0(time_feats, c=self.c)
        time_feats = self.proj(time_feats, c=self.c)

        h_init, c_init = self.init_hidden()
        h_init = pmath_geo.expmap0(h_init, c=self.c)
        h_init = self.proj(h_init, c=self.c)

        c_init = pmath_geo.expmap0(c_init, c=self.c)
        h_init = self.proj(h_init, c=self.c)

        if self.no_time:
            time_feats = torch.ones_like(time_feats).cuda()

        _, (h_out, c_out) = self.rnn(
            sentence_feats, time_feats, (h_init, c_init))
        h_out = pmath_geo.logmap0(h_out, c=self.c)
        h_out = self.proj(h_out, c=self.c)

        c_out = pmath_geo.logmap0(c_out, c=self.c)
        c_out = self.proj(c_out, c=self.c)

        # h_out = self.linear1(h_out)
        # c_out = self.linear2(c_out)

        num_spans, outputs = self.decoder(h_out, c_out)
        return num_spans, outputs


class FullModelV1(nn.Module):
    def __init__(self, input_dim, hid_dim, bs, no_time=False):
        super(FullModelV1, self).__init__()
        self.bs = bs
        self.no_time = no_time
        self.hid_dim = hid_dim
        self.device = torch.device("cuda:0")
        self.rnn = MobiusGRU(input_dim, hid_dim)
        self.c = torch.tensor([1.0]).to(self.device)
        self.tanh = nn.Tanh()
        self.decoder = DecoderGRU(
            input_dim, hid_dim, 4, out_dim=2, bs=self.bs, num_days=10)
        self.linear1 = nn.Linear(hid_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.hid_dim)).to(self.device)
        c = Variable(torch.zeros(self.bs, self.hid_dim)).to(self.device)

        return h, c

    def proj(self, x, c):
        norm = torch.clamp_min(
            x.norm(dim=-1, keepdim=True, p=2), 1e-15)
        maxnorm = (1 - 1e-5) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def forward(self, sentence_feats):
        """
        sentence_feat = (B*75*N)
        time_feats = (B*75)
        """
        sentence_feats = sentence_feats.permute(1, 0, 2)
        sentence_feats = pmath_geo.expmap0(sentence_feats, c=self.c)
        _, h_out = self.rnn(sentence_feats)
        h_out = pmath_geo.logmap0(h_out, c=self.c)
        h_out = h_out.squeeze(0)
        num_spans, outputs = self.decoder(h_out)
        return num_spans, outputs


class DecoderGRU(nn.Module):
    def __init__(
        self, input_dim, hid_dim, num_span_classes, out_dim=2, bs=16, num_days=20
    ):
        super(DecoderGRU, self).__init__()

        self.hid_dim = hid_dim
        self.num_days = num_days
        self.input_dim = input_dim
        self.rnn_cell = nn.GRUCell(input_dim, hid_dim)
        self.fc_in = nn.Linear(hid_dim, input_dim)
        self.fc_out = nn.Linear(hid_dim, out_dim)
        self.num_span_layer = nn.Linear(hid_dim, num_span_classes)
        self.LogSoftmax = nn.LogSoftmax(dim=-1)
        self.Softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, hx):
        """
        hx: (batch_size, hiddem_dim)
        """
        bs, hid_dim = hx.shape
        num_spans = self.Softmax(self.num_span_layer(hx))

        input = torch.zeros(size=(bs, self.input_dim)).cuda()
        outputs = []
        for i in range(self.num_days):
            hx = self.rnn_cell(input, hx)
            input = self.relu(self.fc_in(hx))
            # print(hx.shape)
            outputs.append(self.Softmax(self.fc_out(hx)))

        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute(1, 0, 2)
        # outputs.shape = (batch_size, num_days, 2)
        return num_spans, outputs


class MobiusEncDecGRU(nn.Module):
    def __init__(self, input_dim, hid_dim, num_span_classes, out_dim=2, num_days=10):
        super(MobiusEncDecGRU, self).__init__()
        self.name = "MobiusEncDecGRU"
        self.num_days = num_days
        self.input_dim = input_dim
        self.c = torch.tensor([1.0]).cuda()

        self.enc = MobiusGRU(input_dim, hid_dim)
        self.dec = nn.GRUCell(input_dim, hid_dim)

        self.fc_in = nn.Linear(hid_dim, input_dim)
        self.fc_out = nn.Linear(hid_dim, out_dim)
        self.num_span_layer = nn.Linear(hid_dim, num_span_classes)

        self.LogSoftmax = nn.LogSoftmax(dim=-1)
        self.Softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, sentence_feats):
        """
        sentence_feat = (B*75*N)
        """
        sentence_feats = sentence_feats.permute(1, 0, 2)
        sentence_feats = pmath_geo.expmap0(sentence_feats, c=self.c)
        _, hx = self.enc(sentence_feats)
        hx = pmath_geo.logmap0(hx, c=self.c)
        hx = hx.squeeze(0)

        bs, hid_dim = hx.shape
        num_spans = self.Softmax(self.num_span_layer(hx))

        input = torch.zeros(size=(bs, self.input_dim)).cuda()
        outputs = []
        for i in range(self.num_days):
            hx = self.dec(input, hx)
            input = self.relu(self.fc_in(hx))
            outputs.append(self.Softmax(self.fc_out(hx)))

        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute(1, 0, 2)

        return num_spans, outputs


class MobiusEncDecGRUAttn(nn.Module):
    def __init__(self, input_dim, hid_dim, num_span_classes, maxlen, out_dim=2, num_days=10):
        super(MobiusEncDecGRUAttn, self).__init__()
        self.name = "MobiusEncDecGRUAttn"
        self.num_days = num_days
        self.input_dim = input_dim
        self.c = torch.tensor([1.0]).cuda()

        self.enc = MobiusGRU(input_dim, hid_dim)
        self.dec = nn.GRUCell(input_dim, hid_dim)
        self.attn = SimpleAttn(hid_dim, maxlen=maxlen, use_attention=True)
        self.fc_in = nn.Linear(hid_dim, input_dim)
        self.fc_out = nn.Linear(hid_dim, out_dim)
        self.num_span_layer = nn.Linear(hid_dim, num_span_classes)

        self.LogSoftmax = nn.LogSoftmax(dim=-1)
        self.Softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, sentence_feats, len_feats):
        """
        sentence_feat = (B*75*N)
        len_feat = (B)

        """
        sentence_feats = sentence_feats.permute(1, 0, 2)
        sentence_feats = pmath_geo.expmap0(sentence_feats, c=self.c)
        full, hx = self.enc(sentence_feats)
        hx = pmath_geo.logmap0(hx, c=self.c)
        full = pmath_geo.logmap0(full, c=self.c)
        full = full.permute(1, 0, 2)
        hx = hx.permute(1, 0, 2)
        hx = self.attn(full, hx, len_feats)

        bs, hid_dim = hx.shape
        num_spans = self.Softmax(self.num_span_layer(hx))

        input = torch.zeros(size=(bs, self.input_dim)).cuda()
        outputs = []
        for i in range(self.num_days):
            hx = self.dec(input, hx)
            input = self.relu(self.fc_in(hx))
            outputs.append(self.Softmax(self.fc_out(hx)))

        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute(1, 0, 2)

        return num_spans, outputs
