#coding:utf8
from __future__ import division
import os
from tqdm import tqdm
import time


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import numpy as np
from numpy import zeros, newaxis
from scipy import signal
# M = 60
# T = 30
dB = 15
# L = 20;
# chL = 5;  #多径数
# np.random.seed(123) # use different random seed to get an average performance


def getNum(A):
    pbar = tqdm(total=len(A))
    for k, i in enumerate(A):
        if (i >= -4) and (i < -2):
            A[k] = -3
        elif (i >= -2) and (i < 0):
            A[k] = -1
        elif (i >= 0) and (i < 2):
            A[k] = 1
        elif (i >= 2) and (i < 4):
            A[k] = 3
        # 每次更新进度条的长度
        pbar.update(1)
        # 关闭占用的资源
    pbar.close()
    return A


def generateData(M, T, dB, L, chL, network=None, nonlinear=False):
    EqD = int(round((L+chL)/2))
    # QPSK信号
    # TxS = np.sign(np.random.rand(M) * 2 - 1) + 1j*np.sign(np.random.rand(M) * 2 - 1) #30000
    # 16QAM信道
    M_1 = np.random.rand(M) * 8 - 4
    M_2 = np.random.rand(M) * 8 - 4
    TxS = (getNum(M_1) + 1j*getNum(M_2))/4
    # 信道
    # ch = np.random.randn(chL+1) + 1j * np.random.randn(chL+1)
    ch = [0.0410+0.0109j, 0.0495+0.0123j, 0.0672+0.0170j, 0.0919+0.0235j, 0.7920+0.1281j, 0.3960+0.0871j,
          0.2715+0.0498j, 0.2291+0.0414j, 0.1287+0.0154j, 0.1032+0.0119j]   # 这个是信道的模型吧 H矩阵？？
    ch = ch / np.linalg.norm(ch)
    x = signal.fftconvolve(ch, TxS)[:M]

    #############nonlinear distortions###########
    if nonlinear is True:
        x_abs = np.abs(x)
        x_coefficient = x_abs + 0.2*np.power(x_abs, 2) - 0.1*np.power(x_abs, 3) + 0.5*np.cos(np.pi*x_abs)  # ∣𝑔(𝑣)∣ = ∣𝑣∣ + 0.2*∣𝑣∣^2 − 0.1∣𝑣∣^3 + 0.5*cos(𝜋∣𝑣∣)
        x = x_coefficient * x
    #############nonlinear distortions###########

    # noise
    n = np.random.randn(1, M) + 1j*np.random.randn(1, M)
    n = n/np.linalg.norm(n)*pow(10, (-dB/20))*np.linalg.norm(x)
    x = x + n
    x = x[0]
    K = M-L-1 #19980
    X = []

    for i in range(K):
        X.append(x[i+L+1:i:-1])
    X = np.array(X).T
    test_L = TxS[10+L-EqD+T-10:]  # test labels for cnn
    TxS = TxS[10+L-EqD:10+L-EqD+T-10]
    Y = X[:, 5:T-5]
    test_S = X[:, T-5:]  # test sources for cnn



    # generate training datasets for cnn
    S = []    # training samples
    L = []    # labels
    pbar = tqdm(total=len(TxS))
    for s, l in zip(Y.T, TxS):
        z = zeros((16,))
        S.append([np.array(np.real(s))[:, newaxis], np.array(np.imag(s))[:, newaxis]])
        if np.real(l) == 1/4 and np.imag(l) == 1/4:
            z[0] = 1
            L.append(z)
        elif np.real(l) == 1/4 and np.imag(l) == -1/4:
            z[1] = 1
            L.append(z)
        elif np.real(l) == -1/4 and np.imag(l) == 1/4:
            z[2] = 1
            L.append(z)
        elif np.real(l) == -1/4 and np.imag(l) == -1/4:
            z[3] = 1
            L.append(z)
        elif np.real(l) == 3/4 and np.imag(l) == 1/4:
            z[4] = 1
            L.append(z)
        elif np.real(l) == 3/4 and np.imag(l) == -1/4:
            z[5] = 1
            L.append(z)
        elif np.real(l) == -3/4 and np.imag(l) == 1/4:
            z[6] = 1
            L.append(z)
        elif np.real(l) == -3/4 and np.imag(l) == -1/4:
            z[7] = 1
            L.append(z)
        elif np.real(l) == 1/4 and np.imag(l) == 3/4:
            z[8] = 1
            L.append(z)
        elif np.real(l) == 1/4 and np.imag(l) == -3/4:
            z[9] = 1
            L.append(z)
        elif np.real(l) == -1/4 and np.imag(l) == 3/4:
            z[10] = 1
            L.append(z)
        elif np.real(l) == -1/4 and np.imag(l) == -3/4:
            z[11] = 1
            L.append(z)
        elif np.real(l) == 3/4 and np.imag(l) == 3/4:
            z[12] = 1
            L.append(z)
        elif np.real(l) == 3/4 and np.imag(l) == -3/4:
            z[13] = 1
            L.append(z)
        elif np.real(l) == -3/4 and np.imag(l) == 3/4:
            z[14] = 1
            L.append(z)
        elif np.real(l) == -3/4 and np.imag(l) == -3/4:
            z[15] = 1
            L.append(z)
        # time.sleep(0.00001)
        # 每次更新进度条的长度
        pbar.update(1)
        # 关闭占用的资源
    pbar.close()

    # generate test datasets for cnn
    s = []
    l = []
    pbar = tqdm(total=len(test_L))
    for s1, l1 in zip(test_S.T, test_L):
        z = zeros((16,))
        s.append([np.array(np.real(s1))[:, newaxis], np.array(np.imag(s1))[:, newaxis]])
        if np.real(l1) == 1/4 and np.imag(l1) == 1/4:
            z[0] = 1
            l.append(z)
        elif np.real(l1) == 1/4 and np.imag(l1) == -1/4:
            z[1] = 1
            l.append(z)
        elif np.real(l1) == -1/4 and np.imag(l1) == 1/4:
            z[2] = 1
            l.append(z)
        elif np.real(l1) == -1/4 and np.imag(l1) == -1/4:
            z[3] = 1
            l.append(z)
        elif np.real(l1) == 3/4 and np.imag(l1) == 1/4:
            z[4] = 1
            l.append(z)
        elif np.real(l1) == 3/4 and np.imag(l1) == -1/4:
            z[5] = 1
            l.append(z)
        elif np.real(l1) == -3/4 and np.imag(l1) == 1/4:
            z[6] = 1
            l.append(z)
        elif np.real(l1) == -3/4 and np.imag(l1) == -1/4:
            z[7] = 1
            l.append(z)
        elif np.real(l1) == 1/4 and np.imag(l1) == 3/4:
            z[8] = 1
            l.append(z)
        elif np.real(l1) == 1/4 and np.imag(l1) == -3/4:
            z[9] = 1
            l.append(z)
        elif np.real(l1) == -1/4 and np.imag(l1) == 3/4:
            z[10] = 1
            l.append(z)
        elif np.real(l1) == -1/4 and np.imag(l1) == -3/4:
            z[11] = 1
            l.append(z)
        elif np.real(l1) == 3/4 and np.imag(l1) == 3/4:
            z[12] = 1
            l.append(z)
        elif np.real(l1) == 3/4 and np.imag(l1) == -3/4:
            z[13] = 1
            l.append(z)
        elif np.real(l1) == -3/4 and np.imag(l1) == 3/4:
            z[14] = 1
            l.append(z)
        elif np.real(l1) == -3/4 and np.imag(l1) == -3/4:
            z[15] = 1
            l.append(z)
        # time.sleep(0.05)
        # 每次更新进度条的长度
        pbar.update(1)
        # 关闭占用的资源
    pbar.close()

    if network:
        return S, L, s, l
    return Y.T, TxS.T, x


if __name__ == '__main__':
    X, Y, x, y = generateData(1000, 900, 15, 11, 10, 'cnn')

    X_train_crnn = np.asarray(X)
    Y_train_crnn = np.asarray(Y)
    print("debug")
    print(X_train_crnn)
    print(Y_train_crnn)


# #
# K=M-L; #29980
# X=zeros([L+1,K]);
# # for i in range(1,k+1):
# # 	X(:,i)=X(i+L:-1:i)';
# # end
