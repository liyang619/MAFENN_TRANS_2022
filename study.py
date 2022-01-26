# from data_generator1 import *
# import math
import numpy as np
# #
# # M = 10
# # T = 10
# # # X, Y, x, y = generateData(1000, 900, 15, 11, 10, 'cnn')
# # # print(X)
# # # X_train_crnn = np.asarray(X)
# # # Y_train_crnn = np.asarray(Y)
# # # print(X_train_crnn)
# # # print(Y_train_crnn)
# #
# # # a = []
# # # z = zeros((4,))
# # # z[2] = 1
# # # a.append(z)
# # # print(z, a)
# # #
# # # x = [1, 2, 3]
# # # y = [2, 3, 4]
# # #
# # # if 'cnn':
# # #     print(1)
# # # else:
# # #     print(0)
# # # print(int(round((11+10)/2)))
# # # TxS = np.sign(np.random.rand(10) * 2 - 1) + 1j * np.sign(np.random.rand(10) * 2 - 1)
# # # c = [1,1,1,1,1]
# # # ch = [1, 2, 3, 4, 5]
# # # x = signal.fftconvolve(ch, c)[:M]
# # # print(x)
# # # print(TxS)
# # # h = ch[0:5:2]
# # # # print(ch)
# # # # print(Y)
# # # m = np.random.rand(20)
# # # n = m.reshape(4, -1)
# # # print(n)
# # # n1 = n[1, :]
# # # print(n1)
# #
# # # R = np.random.rand(8)
# # # R = R.reshape(4, -1)
# # # R1 = R[0, :]
# # # R2 = R[1, :]
# # # R3 = R[2, :]
# # # R4 = R[3, :]
# # # a = R1 * R2
# # # print(a, )
# # # print(np.random.randn(1, 5))
# # # print(np.random.randn(5))
# #
# # a = math.sqrt(1/10)
# # a0 =-4.1+3.2j
# # print(np.imag(a))
# # def getSign(x):
# #     b = zeros(4)
# #     if np.real(x) < 0:
# #         b[0] = 1
# #     else:
# #         b[0] = 0
# #     if (np.imag(x) < 2*a) and (np.imag(x) > -2*a):
# #         b[1] = -b[0]
# #     else:
# #         b[1] = b[0]
# #     if np.imag(x) < 0:
# #         b[2] = 1
# #     else:
# #         b[2] = 0
# #     if (np.real(x) < 2*a) and (np.real(x) > -2*a):
# #         b[3] = -b[2]
# #     else:
# #         b[3] = b[2]
# #     return b
# #
# #
# # print(getSign(a0), type(getSign(a0)))
# # print(zeros(4), type(zeros(4)))
# #
# # a = 1
# # b = 2
# # del a
# # a = 3
# # del a, b
# #
# # from tqdm import tqdm
# # import time
# # a = np.random.rand(100)
# # print(a.shape)
# # for i in tqdm(a):
# #   # print(i)
# #   time.sleep(0.1)
# #   pass
# #
# from tqdm import tqdm
# import time
#
# #total参数设置进度条的总长度
# pbar = tqdm(total=100)
# for i in range(100):
#   time.sleep(0.05)
#   #每次更新进度条的长度
#   pbar.update(1)
# #关闭占用的资源
# pbar.close()
dur = 123
with open("./figure/2.txt", "w") as f:
    f.write(str(dur)+"\n")
    # print(text)
f.close()
file_name = "crnn_fb"
l = 123
with open("./figure/{}/time_{}.txt".format(file_name, str(l)), "w") as f:
    f.write(str(dur)+"\n")
f.close()

