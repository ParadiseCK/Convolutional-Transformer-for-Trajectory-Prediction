from Model import *
from torch import optim
from torch.autograd import Variable
from sklearn.externals import joblib
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--c_model', type=int, default=128)
    parser.add_argument('--N', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--s_model', type=int, default=6)
    opt = parser.parse_args()
    model_path = "./checkpoints/model_50.pth"
    c_model = opt.c_model
    s_model = opt.s_model
    N = opt.N
    heads = opt.heads
    transformer = Transformer(c_model,s_model, N, heads)
    transformer.load_state_dict(torch.load(model_path))
    transformer.cuda()
    transformer.eval()
    train_data = joblib.load("./datasets/MOT16-11/train")
    label_data = joblib.load("./datasets/MOT16-11/label")

    train_x = torch.from_numpy(train_data)
    train_y = torch.from_numpy(label_data)
    input = train_x[0]
    label_in = train_y[0, 0, :, :, :, :]
    label_out = train_y[0, 1, :, :, :, :]
    # print(label_out.size())
    input = input.unsqueeze(0)
    label_in = label_in.unsqueeze(0)
    input, label_in = Variable(input).cuda(), Variable(label_in).cuda()
    mask = nopeak_mask(12)
    out = transformer(input, label_in, mask)
    out = out.cpu().data.numpy()
    out = out[0]
    D = []
    X = []
    Y = []
    D_ = []
    X_ = []
    Y_ = []
    for i in range(out.shape[0]):
        d = out[i, 0, 0, 0]
        x = out[i, 1, 0, 0]
        y = out[i, 2, 0, 0]
        d_ = label_out[i, 0, 0, 0]
        x_ = label_out[i, 1, 0, 0]
        y_ = label_out[i, 2, 0, 0]
        D.append(d)
        D_.append(d_)
        X.append(x)
        X_.append(x_)
        Y.append(y)
        Y_.append(y_)
    D = np.array(D)
    X = np.array(X)
    Y = np.array(Y)
    print(X)
    print(Y)
    D_ = np.array(D_)
    X_ = np.array(X_)
    Y_ = np.array(Y_)
    print(X_)
    print(Y_)
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(D * 32, c="r")
    plt.plot(D_ * 32, c="b")
    plt.subplot(1, 2, 2)
    plt.plot(X * 1920, 1080 - Y * 1080, c="r")
    # plt.plot(X_ * 1920, 1080 - Y_ * 1080, c="b")
    plt.show()
