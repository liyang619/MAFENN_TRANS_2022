#coding:utf8
from __future__ import division
import os
from tqdm import tqdm


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import numpy as np
from numpy import zeros, newaxis
from scipy import signal

def generateData(M, T, dB, L, chL, network=None, nonlinear=True):
    EqD = int(round((L+chL)/2))
    # QPSK信号
    TxS = np.sign(np.random.rand(M) * 2 - 1) + 1j*np.sign(np.random.rand(M) * 2 - 1) #30000
    # 信道
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
    K = M-L-1   # 19980
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
        z = zeros((4,))
        S.append([np.array(np.real(s))[:, newaxis], np.array(np.imag(s))[:, newaxis]])
        if np.real(l) == 1 and np.imag(l) == 1:
            z[0] = 1
            L.append(z)
        elif np.real(l) == 1 and np.imag(l) == -1:
            z[1] = 1
            L.append(z)
        elif np.real(l) == -1 and np.imag(l) == 1:
            z[2] = 1
            L.append(z)
        elif np.real(l) == -1 and np.imag(l) == -1:
            z[3] = 1
            L.append(z)
        pbar.update(1)
    pbar.close()

    # generate test datasets for cnn
    s = []
    l = []
    pbar = tqdm(total=len(test_L))
    for s1, l1 in zip(test_S.T, test_L):
        z = zeros((4,))
        s.append([np.array(np.real(s1))[:, newaxis], np.array(np.imag(s1))[:, newaxis]])
        if np.real(l1) == 1 and np.imag(l1) == 1:
            z[0] = 1
            l.append(z)
        elif np.real(l1) == 1 and np.imag(l1) == -1:
            z[1] = 1
            l.append(z)
        elif np.real(l1) == -1 and np.imag(l1) == 1:
            z[2] = 1
            l.append(z)
        elif np.real(l1) == -1 and np.imag(l1) == -1:
            z[3] = 1
            l.append(z)
        pbar.update(1)
    pbar.close()

    if network:
        return S, L, s, l
    return Y.T, TxS.T, x


if __name__ == '__main__':
    X, Y, x, y = generateData(1000, 900, 15, 11, 10, 'cnn', nonlinear=True)

    X_train_crnn = np.asarray(X)
    Y_train_crnn = np.asarray(Y)
    print("debug")
