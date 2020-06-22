import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy
import math
import sys
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class eembedder(nn.Module):
    def __init__(self, c_model):
        super(eembedder, self).__init__()
        self.conv = nn.Conv2d(32, c_model//2, kernel_size=(2, 3), stride=(2, 3), padding=(5, 0))
        self.ban = nn.BatchNorm2d(c_model//2)
        self.conv1 = nn.Conv2d(c_model//2, c_model, kernel_size=(3, 3), stride=1, padding=1)
        self.ban1 = nn.BatchNorm2d(c_model)
    def forward(self, x):
        out = F.relu(self.ban(self.conv(x)))
        out = F.relu(self.ban1(self.conv1(out)))
        return out
class EEmbedder(nn.Module):
    def __init__(self, c_model):
        super(EEmbedder, self).__init__()
        E = []
        for i in range(7):
            E.append(eembedder(c_model))
        self.EE = nn.ModuleList(E)
    def forward(self, x):
        Out = None
        for i in range(x.size(1)):
            if i ==0:
                out = self.EE[i](x[:,i,:,:,: ])
                out = out.unsqueeze(1)
                Out = out
            else:
                out = self.EE[i](x[:, i, :, :, :])
                out = out.unsqueeze(1)
                Out = torch.cat((Out,out), dim= 1)
        return Out

class downsample(nn.Module):
    def __init__(self, c_model):
        super(downsample, self).__init__()
        self.Conv1 = nn.Conv2d(c_model, c_model*2, (3,3), padding=0, stride=1)
        self.bnConv1 = nn.BatchNorm2d(c_model*2)
        self.mpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv2 = nn.Conv2d(c_model*2,c_model*2, (3,3), padding=0, stride=1)
        self.bnConv2 = nn.BatchNorm2d(c_model*2)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = nn.Conv2d(c_model*2, c_model , (3,3), padding=1, stride=1)
    def forward(self, x):
        out = F.relu(self.bnConv1(self.Conv1(x)))
        out = self.mpool1(out)
        out = F.relu(self.bnConv2(self.Conv2(out)))
        out = self.mpool2(out)
        out = self.Conv3(out)
        return out
class downSample(nn.Module):
    def __init__(self, c_model):
        super(downSample, self).__init__()
        D = []
        for i in range(7):
            D.append(downsample(c_model))
        self.DS = nn.ModuleList(D)
    def forward(self, x):
        Out = None
        for i in range(x.size(1)):
            if i == 0:
                out = self.DS[i](x[:, i, :, :, :])
                out = out.unsqueeze(1)
                Out = out
            else:
                out = self.DS[i](x[:, i, :, :, :])
                out = out.unsqueeze(1)
                Out = torch.cat((Out, out), dim=1)
        return Out
class dembedder(nn.Module):
    def __init__(self, c_model):
        super(dembedder, self).__init__()
        self.deConv = nn.ConvTranspose2d(in_channels=3, out_channels=c_model//2, kernel_size=6, stride=1, padding=0)
        self.debanNorm = nn.BatchNorm2d(c_model//2)
        self.deConv1 = nn.Conv2d(c_model//2, c_model, kernel_size=(3, 3), stride=1, padding=1)
        self.debanNorm1= nn.BatchNorm2d(c_model)
    def forward(self, x):
        out = F.relu(self.debanNorm(self.deConv(x)))
        out = F.relu(self.debanNorm1(self.deConv1(out)))
        return out
class DEmbedder(nn.Module):
    def __init__(self, c_model):
        super(DEmbedder, self).__init__()
        D = []
        for i in range(12):
            D.append(dembedder(c_model))
        self.DE = nn.ModuleList(D)
    def forward(self, x):
        Out = None
        for i in range(x.size(1)):
            if i == 0:
                out = self.DE[i](x[:, i, :, :, :])
                out = out.unsqueeze(1)
                Out = out
            else:
                out = self.DE[i](x[:, i, :, :, :])
                out = out.unsqueeze(1)
                Out = torch.cat((Out, out), dim=1)
        return Out
class PositionalEncoder(nn.Module):
    def __init__(self, s_model):
        super().__init__()
        self.s_model = s_model
        pe = torch.zeros(12, s_model)
        for pos in range(12):
            for i in range(0, s_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / s_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / s_model)))
        self.register_buffer('pe', pe)
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.s_model)
        pe_ =None
        for j in range(x.size(1)):
            pe_m = self.pe[j, :]
            pe_m = pe_m.unsqueeze(0)
            # print(pe_m.size())
            for i in range(x.size(3)-1):
                pe_m = torch.cat((pe_m, self.pe[j, :].unsqueeze(0)), dim=0)
            pe_t = pe_m.unsqueeze(0)
            for i in range(x.size(2)-1):
                pe_t = torch.cat((pe_t, pe_m.unsqueeze(0)), dim=0)
            pe_t = pe_t.unsqueeze(0)
            if j ==0:
                pe_= pe_t
            else:
                pe_ = torch.cat((pe_, pe_t), dim=0)
        x = x + Variable(pe_).cuda()
        return x
def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.cuda()
    return np_mask
def attention(Q, K, V, d_k, mask):
    bs = Q[0].size(0)
    sq_l = len(Q)
    chan = Q[0].size(1)
    shape = Q[0].size(2)
    for i, q in enumerate(Q):
        q = q.contiguous().view(q.size(0), -1)
        q = q.unsqueeze(1)
        if i ==0:
            Q_= q
        else:
            Q_ = torch.cat((Q_, q), dim=1)
    for j , k in enumerate(K):
        k = k.contiguous().view(k.size(0), -1)
        k = k.unsqueeze(1)
        if j == 0:
            K_ = k
        else:
            K_ = torch.cat((K_, k), dim=1)
    for l , v in enumerate(V):
        v = v.contiguous().view(v.size(0), -1)
        v = v.unsqueeze(1)
        if l == 0:
            V_ = v
        else:
            V_ = torch.cat((V_, v), dim=1)
    scores = torch.matmul(Q_, K_.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    out = torch.matmul(scores, V_)
    out = out.view(bs, sq_l, chan, shape, shape)
    # sys.exit(0)
    return out
class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, heads, c_model):
        super(EncoderMultiHeadAttention, self).__init__()

        self.c_model = c_model
        self.d_k = c_model // heads
        self.h = heads
        q_conv_list = []
        k_conv_list = []
        v_conv_list = []
        for i in range(7):
            q_conv_list.append(nn.Conv2d(c_model, c_model, kernel_size=(3, 3), stride=1, padding=1))
            k_conv_list.append(nn.Conv2d(c_model, c_model, kernel_size=(3, 3), stride=1, padding=1))
            v_conv_list.append(nn.Conv2d(c_model, c_model, kernel_size=(3, 3), stride=1, padding=1))
        self.q_conv = nn.ModuleList(q_conv_list)
        self.k_conv = nn.ModuleList(k_conv_list)
        self.v_conv = nn.ModuleList(v_conv_list)
        self.out = nn.Conv2d(c_model, c_model, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, q, k, v, mask):
        K, Q, V = [], [], []
        for i in range(q.size(1)):
            a = q[:,i,:,:,:]
            bs = a.size(0)
            shape = (a.size(2), a.size(3))
            # perform linear operation and split into N heads
            q_ = self.q_conv[i](a).view(bs, -1, self.h, shape[0], shape[1])
            # transpose to get dimensions bs * N * sl * d_model
            q_ = q_.transpose(1, 2)
            Q.append(q_)
        for i in range(k.size(1)):
            a = k[:,i,:,:,:]
            bs = a.size(0)
            shape = (a.size(2), a.size(3))
            # perform linear operation and split into N heads
            k_ = self.k_conv[i](a).view(bs, -1, self.h, shape[0], shape[1])
            # transpose to get dimensions bs * N * sl * d_model
            k_ = k_.transpose(1, 2)
            K.append(k_)
        for i in range(v.size(1)):
            a = v[:,i,:,:,:]
            bs = a.size(0)
            shape = (a.size(2), a.size(3))
            # perform linear operation and split into N heads
            v_ = self.v_conv[i](a).view(bs, -1, self.h, shape[0], shape[1])
            # transpose to get dimensions bs * N * sl * d_model
            v_ = v_.transpose(1, 2)
            V.append(v_)
        for m in range(self.h):
            Q_, K_, V_ = [], [], []
            for n in range(len(Q)):
                Q_.append(Q[n][:, m, :, :, :])
            for n in range(len(K)):
                K_.append(K[n][:, m, :, :, :])
            for n in range(len(V)):
                V_.append(V[n][:, m, :, :, :])
            out = attention(Q_, K_, V_, self.d_k, mask)
            if m ==0:
                Out = out
            else:
                Out = torch.cat((Out, out), dim=2)
        return Out


class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, heads, c_model):
        super(DecoderMultiHeadAttention, self).__init__()

        self.c_model = c_model
        self.d_k = c_model // heads
        self.h = heads
        q_conv_list = []
        k_conv_list = []
        v_conv_list = []
        for i in range(12):
            q_conv_list.append(nn.Conv2d(c_model, c_model, kernel_size=(3, 3), stride=1, padding=1))
            k_conv_list.append(nn.Conv2d(c_model, c_model, kernel_size=(3, 3), stride=1, padding=1))
            v_conv_list.append(nn.Conv2d(c_model, c_model, kernel_size=(3, 3), stride=1, padding=1))
        self.q_conv = nn.ModuleList(q_conv_list)
        self.k_conv = nn.ModuleList(k_conv_list)
        self.v_conv = nn.ModuleList(v_conv_list)
        self.out = nn.Conv2d(c_model, c_model, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, q, k, v, mask):
        K, Q, V = [], [], []
        for i in range(q.size(1)):
            a = q[:,i,:,:,:]
            bs = a.size(0)
            shape = (a.size(2), a.size(3))
            # perform linear operation and split into N heads
            q_ = self.q_conv[i](a).view(bs, -1, self.h, shape[0], shape[1])
            # transpose to get dimensions bs * N * sl * d_model
            q_ = q_.transpose(1, 2)
            Q.append(q_)
        for i in range(k.size(1)):
            a = k[:,i,:,:,:]
            bs = a.size(0)
            shape = (a.size(2), a.size(3))
            # perform linear operation and split into N heads
            k_ = self.k_conv[i](a).view(bs, -1, self.h, shape[0], shape[1])
            # transpose to get dimensions bs * N * sl * d_model
            k_ = k_.transpose(1, 2)
            K.append(k_)
        for i in range(v.size(1)):
            a = v[:,i,:,:,:]
            bs = a.size(0)
            shape = (a.size(2), a.size(3))
            # perform linear operation and split into N heads
            v_ = self.v_conv[i](a).view(bs, -1, self.h, shape[0], shape[1])
            # transpose to get dimensions bs * N * sl * d_model
            v_ = v_.transpose(1, 2)
            V.append(v_)
        for m in range(self.h):
            Q_, K_, V_ = [], [], []
            for n in range(len(Q)):
                Q_.append(Q[n][:, m, :, :, :])
            for n in range(len(K)):
                K_.append(K[n][:, m, :, :, :])
            for n in range(len(V)):
                V_.append(V[n][:, m, :, :, :])
            out = attention(Q_, K_, V_, self.d_k, mask)
            if m ==0:
                Out = out
            else:
                Out = torch.cat((Out, out), dim=2)
        return Out


class FeedForward(nn.Module):
    def __init__(self, c_model):
        super(FeedForward, self).__init__()
        self.conv = nn.Conv2d(c_model, c_model, kernel_size=(3, 3), stride=1, padding=1)
    def forward(self, x):
        x = F.relu(self.conv(x))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, c_model, heads):
        super(EncoderLayer, self).__init__()
        self.norm_1 = nn.BatchNorm2d(c_model)
        self.norm_2 = nn.BatchNorm2d(c_model)
        self.attn = EncoderMultiHeadAttention(heads, c_model)
        self.ff = FeedForward(c_model)

    def forward(self, x):
        for i in range(x.size(1)):
            if i ==0:
                x2 = self.norm_1(x[:,i,:,:,:]).unsqueeze(1)
            else:
                x2 = torch.cat((x2, self.norm_1(x[:,i,:,:,:]).unsqueeze(1)), dim=1)
        x2 = self.attn(x2,x2,x2, None)
        for j in range(x.size(1)):
            x_ = x[:,i,:,:,:] + x2[:,i,:,:,:]
            x2_ = self.norm_2(x_)
            x_ = x_ + self.ff(x2_)
            if j ==0:
                out = x_.unsqueeze(1)
            else:
                out = torch.cat((out, x_.unsqueeze(1)),dim=1)
        return out
class Encoder(nn.Module):
    def __init__(self, c_model, s_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = EEmbedder(c_model)
        self.pe = PositionalEncoder(s_model)
        self.downSample = downSample(c_model)
        self.layers = get_clones(EncoderLayer(c_model, heads), N)
    def forward(self, src):
        x = self.embed(src)
        x = self.downSample(x)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self, c_model, heads):
        super(DecoderLayer, self).__init__()
        self.norm_1 = nn.BatchNorm2d(c_model)
        self.norm_2 = nn.BatchNorm2d(c_model)
        self.norm_3 = nn.BatchNorm2d(c_model)
        self.attn1 = DecoderMultiHeadAttention(heads, c_model)
        self.attn2 = DecoderMultiHeadAttention(heads, c_model)
        self.ff = FeedForward(c_model)
    def forward(self, x, e_outs, mask):

        for i in range(x.size(1)):
            if i ==0:
                x2 = self.norm_1(x[:,i,:,:,:]).unsqueeze(1)
            else:
                x2 = torch.cat((x2, self.norm_1(x[:,i,:,:,:]).unsqueeze(1)), dim=1)
        x2 = self.attn1(x2, x2, x2, mask)
        for j in range(x.size(1)):
            x_ = x[:,i,:,:,:] + x2[:,i,:,:,:]
            x2_ = self.norm_2(x_)
            if j ==0:
                out = x2_.unsqueeze(1)
            else:
                out = torch.cat((out, x2_.unsqueeze(1)),dim=1)
        x3 = self.attn2(out,e_outs, e_outs, None)
        for j in range(x.size(1)):
            x_ = x[:,i,:,:,:] + x3[:,i,:,:,:]
            x3_ = self.norm_3(x_)
            if j ==0:
                out_ = x3_.unsqueeze(1)
            else:
                out_ = torch.cat((out_, x3_.unsqueeze(1)),dim=1)
        for j in range(x3.size(1)):
            x__ = x[:,i,:,:,:] + x3[:,i,:,:,:]
            x__ = x__ + self.ff(x__)
            if j ==0:
                out__ = x__.unsqueeze(1)
            else:
                out__ = torch.cat((out__, x__.unsqueeze(1)),dim=1)
        return out__
class Decoder(nn.Module):
    def __init__(self, c_model, s_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.embed = DEmbedder(c_model)
        self.pe = PositionalEncoder(s_model)
        self.layers = get_clones(DecoderLayer(c_model, heads), N)
        self.conv1 = nn.Conv2d(c_model, c_model//2, kernel_size=(3, 3), stride=1, padding=1)
        self.banNorm1 = nn.BatchNorm2d(c_model//2)
        self.conv2 = nn.Conv2d(in_channels=c_model//2, out_channels=3, kernel_size=s_model, stride=1, padding=0)
        self.banNorm2 = nn.BatchNorm2d(3)
    def forward(self, src, eouts, mask):
        x = self.embed(src)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, eouts, mask)
        for j in range(x.size(1)):
            out = F.relu(self.banNorm1(self.conv1(x[:,j,:,:,:])))
            out = F.relu(self.banNorm2(self.conv2(out)))
            if j == 0:
                Out = out.unsqueeze(1)
            else:
                Out = torch.cat((Out,out.unsqueeze(1)), dim=1)
        return Out
class Transformer(nn.Module):
    def __init__(self, c_model, s_model, N, heads,):
        super(Transformer, self).__init__()
        self.encoder = Encoder(c_model, s_model, N, heads)
        self.decoder = Decoder(c_model, s_model, N, heads)
    def forward(self, src, trg, mask):
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, mask)
        return d_output
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
if __name__ == "__main__":
    batch_size = 2
    inp_chans = 32
    shape = [54, 96]
    src = Variable(torch.rand(batch_size, 7, inp_chans, shape[0], shape[1])).cuda()
    trc = Variable(torch.rand(batch_size, 12, 3, 1, 1)).cuda()
    c_model = 128
    s_model = 6
    N = 6
    heads = 8
    mask = nopeak_mask(12)
    print("EncoderInputSize:{}".format(src.size()))
    print("DecoderInputSize:{}".format(trc.size()))
    model = Transformer(c_model,s_model, N, heads)
    model.cuda()
    model.apply(weigth_init)
    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of params: %.2fM" % (total / 1e6))
    out = model(src, trc, mask)
    print("OutSize:{}".format(out.size()))