import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from new_data_generator import *
import datetime

control_training = False
train_num = 50000
all_num = 100000
showing = False
# showing = True
file_name = 'pre_RLS'

start_time = datetime.datetime.now()
L = 11
chL = 10
EqD = int(round((L+chL)/2))
acc_list = []
dB = range(0, 31, 2)
for db in dB:
    if control_training:
        if db == 14:
            train_num = 21300
            all_num = 50000
        elif db == 12:
            train_num = 22000
            all_num = 50000
        elif db == 10:
            train_num = 23200
            all_num = 80000
        elif db == 8:
            train_num = 25100
            all_num = 60000
        elif db == 6:
            train_num = 27945
            all_num = 50000
        elif db == 4:
            train_num = 32100
            all_num = 100000
        elif db == 2:
            train_num = 38890
            all_num = 100000
        elif db == 0:
            train_num = 49150
            all_num = 100000
        elif db == 16:
            train_num = 21300
            all_num = 100000
        elif db == 18:
            train_num = 10000
            all_num = 10000000
    else:
        train_num = 50000
        all_num = 500000
    X_train, Y_train, X_test, Y_test, DiscreteY_train, DiscreteY_test = generateData(all_num, train_num, db, L, chL, 'cnn')
    X_train = np.asarray(X_train).T
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test).T
    Y_test = np.asarray(Y_test)
    DiscreteY_train = np.asarray(DiscreteY_train)
    DiscreteY_test = np.asarray(DiscreteY_test)
    c = np.zeros((1, L+1))
    R_inverse = 100*np.eye(L+1)
    for k in range(X_train.shape[1]):
        n1 = Y_train[k]
        n2 = c.dot(X_train[:, k])
        e = n1 - n2
        print(e)
        filtered_infrmn_vect = R_inverse.dot(X_train[:, k])  # (13,1)
        norm_error_power = np.conj(X_train[:, k].T).dot(filtered_infrmn_vect)
        gain_constant = 1 / (1 + norm_error_power)
        norm_filtered_infrmn_vect = gain_constant * np.conj(filtered_infrmn_vect.T)
        c = c + e * norm_filtered_infrmn_vect
        R_inverse = R_inverse - np.conj(norm_filtered_infrmn_vect.reshape((12, 1))).dot(norm_filtered_infrmn_vect.reshape((1, 12)))

    # 得到经过修正后的数据
    sb = np.dot(c, X_test)
    sb = sb.T
    # 计算数据到四个星座点的距离
    # class_1 = np.array([[1+1j]])
    # class_2 = np.array([[1-1j]])
    # class_3 = np.array([[-1+1j]])
    # class_4 = np.array([[-1-1j]])
    # distance_1 = -1*abs(sb - class_1)
    # distance_2 = -1*abs(sb - class_2)
    # distance_3 = -1*abs(sb - class_3)
    # distance_4 = -1*abs(sb - class_4)
    class_1 = np.array([[1+1j]])
    class_2 = np.array([[1-1j]])
    class_3 = np.array([[-1+1j]])
    class_4 = np.array([[-1-1j]])
    class_5 = np.array([[3+1j]])
    class_6 = np.array([[3-1j]])
    class_7 = np.array([[-3+1j]])
    class_8 = np.array([[-3-1j]])
    class_9 = np.array([[1+3j]])
    class_10 = np.array([[1-3j]])
    class_11 = np.array([[-1+3j]])
    class_12 = np.array([[-1-3j]])
    class_13 = np.array([[3+3j]])
    class_14 = np.array([[3-3j]])
    class_15 = np.array([[-3+3j]])
    class_16 = np.array([[-3-3j]])
    distance_1 = -1*abs(sb - class_1)
    distance_2 = -1*abs(sb - class_2)
    distance_3 = -1*abs(sb - class_3)
    distance_4 = -1*abs(sb - class_4)
    distance_5 = -1*abs(sb - class_5)
    distance_6 = -1*abs(sb - class_6)
    distance_7 = -1*abs(sb - class_7)
    distance_8 = -1*abs(sb - class_8)
    distance_9 = -1*abs(sb - class_9)
    distance_10 = -1*abs(sb - class_10)
    distance_11 = -1*abs(sb - class_11)
    distance_12 = -1*abs(sb - class_12)
    distance_13 = -1*abs(sb - class_13)
    distance_14 = -1*abs(sb - class_14)
    distance_15 = -1*abs(sb - class_15)
    distance_16 = -1*abs(sb - class_16)
    k = np.concatenate((distance_1, distance_2, distance_3, distance_4, distance_5, distance_6, distance_7, distance_8,
                        distance_9, distance_10, distance_11, distance_12, distance_13, distance_14, distance_15, distance_16), axis=1)
    index = k.argmax(axis=1)
    total = DiscreteY_test.shape[0]
    correct = (index == DiscreteY_test.argmax(axis=1))
    correct = correct.sum()
    acc_list.append(correct/total)
    print("Accuracy is:{}".format(correct/total))
    print("Accuracy in {}dB is: {}".format(db, acc_list))
    np.savetxt("./acc_every_db.txt", np.array(acc_list), delimiter=',')
    if showing:
        plt.rcParams['font.size'] = 14
        plt.rcParams['lines.linewidth'] = 4.0
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1, title=u"发送序列")
        ax1.scatter(Y_test.real, Y_test.imag)
        ax2 = fig.add_subplot(2, 2, 2, title=u"接收序列")
        ax2.scatter(X_test.real, X_test.imag)
        ax3 = fig.add_subplot(2, 2, 3, title=u"恢复序列")
        ax3.scatter(sb.real, sb.imag)
        plt.show()

stop_time = datetime.datetime.now()
duration = stop_time - start_time
print("Programme has finished. Totally consumed:{}".format(duration))
