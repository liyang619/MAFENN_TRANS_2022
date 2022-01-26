import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import visdom
import argparse
import torch.nn.functional as F
import random
import numpy as np
import os
from data_generator import *
from utils import make_dir
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime


# 这个版本依然是获取过去的反馈信号的，但是是把数组分开，依次从中间开始抽取数据
# this version will respectively fetch the data from the dataset not in order
# 8999990  999993 
# 999990   2 * 3 * 3 * 5 * 41 * 271
# 8999990  2 * 5 * 397 * 2267     1985 * 4534
# all feedback will be processed by the convert matrix 
# 9次反馈的版本号
# 这个版本的反馈，需要通过将之前的判决输出反馈回来
# 对rcnn_fb_6版本的代码进行了修改与整理，使得代码风格更为的简洁明亮
start_time = datetime.datetime.now()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
file_name = "nonlinear_rcnn_nnfb_5s_3roll"
fb_length = 5
dB = range(30, -1, -2)
load_model = True
cuda_enable = torch.cuda.is_available()
learn_rate = 1e-3
assist_lr = 1e-3
lossFunction = {
    'l1': nn.SmoothL1Loss(),
    'l2': nn.MSELoss(),
    'cross_entropy': nn.CrossEntropyLoss(),
    'bce': nn.BCELoss(),
}
epoch = 50
top_num = 4
loss_epoch = 0
loss_iter_test = []  # 记录每个epoch的错误率
loss_iter_train = []  # 记录每个epoch训练时候的错误率
train_acc = [] # 绘制训练正确率
test_acc = []  # 绘制测试正确率
x_axis = []  # 绘图用的横坐标
train_idx = 0
test_idx = 0
acc_in_dbs = []
best_acc = 0

# 含有所有的Loss的List
loss_list_all = []
loss_decay = 0.8
convert = torch.Tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]]).float()
if cuda_enable:
    convert = convert.cuda()
# index_train = np.linspace(0,8999990-4534,1985).astype(np.int32) # 1985 * 4534 = batch_size * batch_idx
index_train = np.linspace(0,8999990-11335,794).astype(np.int32) # 794 * 11335 = batch_size * batch_idx
index_test = np.linspace(0,999990-8130,123).astype(np.int32) # 123 * 8130 = batch_size * batch_idx
train_idx = 11335
train_batch_size = 794
test_idx = 8130
test_batch_size = 123


class Equalizer(nn.Module):
    def __init__(self):
        super(Equalizer, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, (3, 3), 1, (1, 1))
        self.conv_2 = nn.Conv2d(32, 32, (2, 3), 1, (0, 1))
        self.pool_1 = nn.MaxPool2d((1, 2), stride=2)
        # self.pool_1 = nn.MaxPool2d((2, 2), stride=2)
        self.lstm_1 = nn.LSTM(
            input_size=32,
            hidden_size=128,  # rnn hidden unit
            num_layers=2,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        
        # self.medium = nn.Linear((6+fb_length)*16, 128)
        self.predict_1 = nn.Linear(128, 4)
        self.sfm = nn.Softmax(dim=1)

    def forward(self, data, h_state):
        data = self.conv_1(data)
        data = F.relu(data)
        data = self.conv_2(data)
        data = F.relu(data)
        data = self.pool_1(data)
        data = data.view(-1, math.floor((fb_length+12+1)/2), 32) # 反馈网络，这里需要被改动
        if h_state is None:
            r_out, h_state = self.lstm_1(data, None)
        else:
            r_out, h_state = self.lstm_1(data, (h_state[0].detach(), h_state[1].detach()))
        data = r_out[:, -1, :]
        data = F.relu(data)
        output = self.predict_1(data)
        output = self.sfm(output)
        #return output, h_state, fb
        return output, data

class Assistor(nn.Module):
    def __init__(self):
        super(Assistor, self).__init__()
        self.nn1 = nn.Linear(128, 32)
        self.nn2 = nn.Linear(32, 2)

    def forward(self, data):
        data = self.nn1(data)
        data = F.relu(data)
        out = self.nn2(data)
        out = torch.tanh(out)
        return out


def drawfigure(x,trainloss,testlossmse,testlossce,testacc,path):
    plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    plt.sca(ax1)
    plt.title('train_loss')
    plt.plot(x, trainloss, color='red')
    plt.tight_layout()  # swb加的
    plt.sca(ax2)
    plt.title('train_acc')
    plt.plot(x, testlossmse, color='red')
    plt.tight_layout()  # swb加的
    plt.sca(ax3)
    plt.title('test_loss')
    plt.plot(x, testlossce, color='red')
    plt.tight_layout()  # swb加的
    plt.sca(ax4)
    plt.title('test_acc')
    plt.plot(x, testacc, color='blue')
    plt.tight_layout()  # swb加的
    plt.savefig("{}".format(path), dpi=500)
    plt.close()


for db in dB:
    # build the network and the optimizer
    equal = Equalizer()
    assist = Assistor()
    if load_model:
        resume = './figure/{}/equal_{}db.pt'.format(file_name, db)
        name = "Equalizer"
        name_2 = "Assistor"
        if os.path.isfile(resume):
            print("     ==> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            for n, weights in checkpoint[name].items():
                if len(weights.size()) == 0:
                    checkpoint[name][n] = checkpoint[name][n].reshape(1)
            for n, weights in checkpoint[name_2].items():
                if len(weights.size()) == 0:
                    checkpoint[name_2][n] = checkpoint[name_2][n].reshape(1)
            # 加载均衡器
            pretrained_dict = checkpoint[name]
            model_dict = equal.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            equal.load_state_dict(model_dict)
            # 加载反馈网络部分
            pretrained_dict = checkpoint[name_2]
            model_dict = assist.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            assist.load_state_dict(model_dict)
            print("     ==> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("     ==> no checkpoint found at '{}'".format(resume))
    optimizer = optim.Adam(equal.parameters(), lr=learn_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    assist_optim = optim.Adam(assist.parameters(), lr=assist_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    if cuda_enable:
        equal = equal.cuda()
        assist = assist.cuda()
    # 读取训练数据
    X_train_crnn = np.load("./database_nonlinear/{}_train_x.npy".format(db)) #读取文件
    Y_train = np.load("./database_nonlinear/{}_train_y.npy".format(db)) #读取文件
    X_test_crnn = np.load("./database_nonlinear/{}_test_x.npy".format(db)) #读取文件
    Y_test = np.load("./database_nonlinear/{}_test_y.npy".format(db)) #读取文件
    # get the top 999990 data of the whole dateset
    X_test_crnn = X_test_crnn[0:999990, :,:,:]
    Y_test = Y_test[0:999990, :]
    # 这里准备数据
    X_train_crnn = torch.from_numpy(X_train_crnn)
    Y_train = torch.from_numpy(Y_train)
    X_test_crnn = torch.from_numpy(X_test_crnn)
    Y_test = torch.from_numpy(Y_test)
    
    # 初始化数组，用以画图
    loss_iter_test = []  # 记录每个epoch的错误率
    loss_iter_train = []  # 记录每个epoch训练时候的错误率
    train_acc = [] # 绘制训练正确率
    test_acc = []  # 绘制测试正确率
    x_axis = []  # 绘图用的横坐标
    fb_buffer = [] # 缓存反馈回来的变量
    best_acc = 0

    for i in range(1, epoch):
        # update the learning rate
        if i == 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learn_rate/5
            for param_group in assist_optim.param_groups:
                param_group['lr'] = assist_lr/5
            print("==> LR now:{} ".format(param_group['lr']))
        if i == 3:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learn_rate/10
            for param_group in assist_optim.param_groups:
                param_group['lr'] = assist_lr/10
            print("==> LR now:{} ".format(param_group['lr']))
        # start train the network
        equal.train()
        assist.train()
        total = 0
        correct = 0
        fb_buffer = []
        for fb_buffer_count in range(fb_length+1):
            fb_buffer.append(torch.zeros((train_batch_size, 1, 2, 1)).to("cuda"))
        # train_idx = 25000
        for batch_idx in range(train_idx):
            # prepare the feedback buffer
            raw_inputs0 = X_train_crnn[index_train+batch_idx, :, :, :].float()
            targets0 = Y_train[index_train+batch_idx, :].long()
            if cuda_enable:
                raw_inputs0, targets0 = raw_inputs0.to("cuda"), targets0.to("cuda")
            optimizer.zero_grad()
            assist_optim.zero_grad()
            loss = 0
            # feedback
            result, latent = equal(torch.cat((raw_inputs0, torch.cat(fb_buffer, 3)), 3), None)
            loss_1 = nn.NLLLoss()(torch.log(result), torch.topk(targets0, 1)[1].squeeze(1))
            feedback = assist(latent)
            loss_fd_1 = lossFunction['l2'](feedback, targets0.float().mm(convert))
            result_1 = feedback.view(result.size(0), 1, 2, 1)
            fb_buffer[0] = result_1

            result, latent = equal(torch.cat((raw_inputs0, torch.cat(fb_buffer, 3)), 3), None)
            loss_2 = nn.NLLLoss()(torch.log(result), torch.topk(targets0, 1)[1].squeeze(1))
            feedback = assist(latent)
            loss_fd_2 = lossFunction['l2'](feedback, targets0.float().mm(convert))
            result_2 = feedback.view(result.size(0), 1, 2, 1)
            fb_buffer[0] = result_2

            result, latent = equal(torch.cat((raw_inputs0, torch.cat(fb_buffer, 3)), 3), None)
            loss_3 = nn.NLLLoss()(torch.log(result), torch.topk(targets0, 1)[1].squeeze(1))
            feedback = assist(latent)
            loss_fd_3 = lossFunction['l2'](feedback, targets0.float().mm(convert))
            result_3 = feedback.view(result.size(0), 1, 2, 1).clone().detach()
            fb_buffer.insert(1, result_3)
            fb_buffer.pop()
            fb_buffer[0] = torch.zeros((train_batch_size, 1, 2, 1)).cuda()
            # update the Equalizer's parameters
            loss = loss_1 + loss_2 + loss_3 + loss_fd_1 + loss_fd_2 + loss_fd_3
            loss.backward()
            optimizer.step()
            assist_optim.step()
            loss_epoch += loss.item()
            _, predicted = result.max(1)
            total += targets0.size(0)
            correct += predicted.eq(torch.topk(targets0, 1)[1].squeeze(1)).sum().item()
            if batch_idx % 50 == 0:
                print("  train   db:{};epoch:{};({}/{});loss:{};acc:{};".format(db, i, batch_idx, train_idx, loss_epoch/(batch_idx+1), correct/total), end='\r')
        
        train_acc += [correct/total]
        loss_iter_train += [loss_epoch/(batch_idx+1)]
        loss_epoch = 0
        total = 0
        correct = 0
        equal.eval()
        assist.eval()
        fb_buffer = []
        for fb_buffer_count in range(fb_length+1):
            fb_buffer.append(torch.zeros((test_batch_size, 1, 2, 1)).to("cuda"))

        with torch.no_grad():
            for batch_idx in range(test_idx):
                 # prepare the feedback buffer
                raw_inputs0 = X_test_crnn[index_test+batch_idx, :, :, :].float()
                targets0 = Y_test[index_test+batch_idx, :].long()
                if cuda_enable:
                    raw_inputs0, targets0 = raw_inputs0.to("cuda"), targets0.to("cuda")
                loss = 0
                # feedback
                result, latent = equal(torch.cat((raw_inputs0, torch.cat(fb_buffer, 3)), 3), None)
                loss_1 = nn.NLLLoss()(torch.log(result), torch.topk(targets0, 1)[1].squeeze(1))
                feedback = assist(latent)
                loss_fd_1 = lossFunction['l2'](feedback, targets0.float().mm(convert))
                result_1 = feedback.view(result.size(0), 1, 2, 1)
                fb_buffer[0] = result_1

                result, latent = equal(torch.cat((raw_inputs0, torch.cat(fb_buffer, 3)), 3), None)
                loss_2 = nn.NLLLoss()(torch.log(result), torch.topk(targets0, 1)[1].squeeze(1))
                feedback = assist(latent)
                loss_fd_2 = lossFunction['l2'](feedback, targets0.float().mm(convert))
                result_2 = feedback.view(result.size(0), 1, 2, 1)
                fb_buffer[0] = result_2

                result, latent = equal(torch.cat((raw_inputs0, torch.cat(fb_buffer, 3)), 3), None)
                loss_3 = nn.NLLLoss()(torch.log(result), torch.topk(targets0, 1)[1].squeeze(1))
                feedback = assist(latent)
                loss_fd_3 = lossFunction['l2'](feedback, targets0.float().mm(convert))
                result_3 = feedback.view(result.size(0), 1, 2, 1).clone().detach()
                fb_buffer.insert(1, result_3)
                fb_buffer.pop()
                fb_buffer[0] = torch.zeros((test_batch_size, 1, 2, 1)).cuda()
                # update the Equalizer's parameters
                loss = loss_1 + loss_2 + loss_3 + loss_fd_1 + loss_fd_2 + loss_fd_3
                _, predicted = result.max(1)
                total += targets0.size(0)
                correct += predicted.eq(torch.topk(targets0, 1)[1].squeeze(1)).sum().item()
                if batch_idx % 50 == 0:
                    print("  eval    db:{};epoch:{};({}/{});loss:{};acc:{}".format(db, i, batch_idx, test_idx, loss_epoch/(batch_idx+1), correct/total), end='\r')
        print(" {} epoch now. test acc in {}dB is:{}".format(i, db, correct/total))
        x_axis += [i]
        loss_iter_test += [loss_epoch/(batch_idx+1)]
        test_acc += [correct/total]
        if test_acc[-1] >= best_acc:
            make_dir("./figure/{}/".format(file_name))
            print('      ==>Saving..') 
            state = {
                'Equalizer': equal.state_dict(),
                'Assistor': assist.state_dict(),
                'epoch': i,
                'dB': db,
            }
            path = './figure/{}/equal_{}db.pt'.format(file_name, db)
            torch.save(state, path)
            print('      ==>Saved '+path)
            best_acc = test_acc[-1]
        make_dir("./figure/{}/".format(file_name))
        drawfigure(x_axis, loss_iter_train, train_acc, loss_iter_test, test_acc, './figure/{}/acc_{}.png'.format(file_name, db))
    # print("The each acc of every epoch in {}db is:{} ".format(db, test_acc))
    for pop_counter in range(top_num):
        test_acc.pop(0)
    acc_in_dbs += [np.mean(test_acc)]
    print("accurarcy is respectively: ", acc_in_dbs)
    np.savetxt("./figure/{}/acc_every_db.txt".format(file_name), np.array(acc_in_dbs), delimiter=',')

stop_time = datetime.datetime.now()
duration = stop_time - start_time
print("Programme has finished. Totally consumed:{}".format(duration))







