from Model import *
from torch import optim
from torch.autograd import Variable
from sklearn.externals import joblib
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import argparse
import datetime
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lrDecay', type=int, default=10)
    parser.add_argument('--c_model', type=int, default=128)
    parser.add_argument('--N', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--s_model', type=int, default=6)
    parser.add_argument('--lr', default=0.001)
    opt = parser.parse_args()
    model_path = "./checkpoints/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    batch_size = opt.batchSize
    print("batchSize:{}".format(batch_size))
    c_model = opt.c_model
    s_model = opt.s_model
    N = opt.N
    heads = opt.heads
    mask = nopeak_mask(12)
    transformer = Transformer(c_model,s_model, N, heads)
    transformer.apply(weigth_init)
    # print(transformer)
    transformer.cuda()
    transformer.train()
    # optimizer = optim.SGD(transformer.parameters(), lr=opt.lr, momentum=0.9)
    optimizer = optim.Adam(transformer.parameters(), lr=opt.lr)
    criterion = torch.nn.SmoothL1Loss(reduction="mean")
    loss = None
    lrDecay = opt.lrDecay
    print("lrDecay:{}".format(lrDecay))
    train_data = joblib.load("./datasets/MOT16-11/train")
    label_data = joblib.load("./datasets/MOT16-11/label")
    train_x = torch.from_numpy(train_data)
    train_y = torch.from_numpy(label_data)

    print(train_x.size(), train_y.size())
    dataSet = TensorDataset(train_x, train_y)

    train_loader = DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=True, num_workers=2)
    total_epoch = opt.epochs
    print("epochs:{}".format(total_epoch))
    Loss = []
    for epoch in range(total_epoch):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            t_lables = labels[:, 0, :, :, :, :]
            l_lables = labels[:, 1, :, :, :, :]
            inputs, t_lables, l_lables = Variable(inputs).cuda(), Variable(t_lables).cuda(), Variable(l_lables).cuda()
            out = transformer(inputs, t_lables, mask)
            loss = criterion(out, l_lables)
            optimizer.zero_grad()
            # loss.backward(torch.ones_like(loss))
            loss.backward()
            optimizer.step()
            print('time:{:%Y-%m-%d %X} , epoch:{} , batch:{}, loss is: {:.3f}'.format(datetime.datetime.now(), str(epoch + 1), str(i + 1), loss.item()))
        Loss.append(loss.item())
        if (epoch + 1) % lrDecay == 0:
            lr = optimizer.param_groups[0]['lr'] * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            torch.save(transformer.state_dict(), model_path + "model_" + str(epoch + 1) + ".pth")
            print("Saved The Model and change lr to:{}".format(lr))
    np.savetxt(model_path + "loss.txt", np.array(Loss))