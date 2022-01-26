#coding:utf8
from __future__ import division
import numpy as np
from numpy import zeros, newaxis
from scipy import signal
M = 60
T = 30
dB = 25;
L = 20;
chL = 5;  #多径数

np.random.seed(123)

def generateData(M,T,dB,L,chL,network=None):
    EqD = int(round((L+chL)/2))
    #QPSK信号
    TxS = np.sign(np.random.rand(M) * 2 - 1) + 1j*np.sign(np.random.rand(M) * 2 - 1) #30000
    #信道
    #ch = np.random.randn(chL+1) + 1j * np.random.randn(chL+1)
    ch = [0.0410+0.0109j,0.0495+0.0123j,0.0672+0.0170j,0.0919+0.0235j,
     0.7920+0.1281j,0.3960+0.0871j,0.2715+0.0498j,0.2291+0.0414j,0.1287+0.0154j,
     0.1032+0.0119j]
    ch = ch / np.linalg.norm(ch)
    x = signal.fftconvolve(ch,TxS)[:M]
    #noise
    n=np.random.randn(1,M)+1j*np.random.randn(1,M);
    n=n/np.linalg.norm(n)*pow(10,(-dB/20))*np.linalg.norm(x);
    x = x + n
    x = x[0]
    K = M-L-1 #19980
    X = []

    for i in range(K):
        X.append(x[i+L+1:i:-1])
    X = np.array(X).T
    test_L = TxS[10+L-EqD+T-10:]  # test labels for cnn
    TxS = TxS[10+L-EqD:10+L-EqD+T-10]
    Y = X[:,5:T-5]
    test_S = X[:, T-5:]  # test sources for cnn



    # generate training datasets for cnn
    S = []    # training samples
    L = []    # labels
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

    # generate test datasets for cnn
    s = []
    l = []
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

    if network:
        return S, L, s, l
    return Y.T,TxS.T,x

if __name__ == '__main__':
    X, Y, x, y = generateData(M,T,dB,L,chL,network='cnn')



# #
# K=M-L; #29980
# X=zeros([L+1,K]);
# # for i in range(1,k+1):
# # 	X(:,i)=X(i+L:-1:i)';
# # end
