import os
from data_generator_qpsk import *
import numpy as np
import time


def make_dir(path):
    if not os.path.isdir(path):
        print('     Create dir:{}...'.format(path))
        os.mkdir(path)


def gen_save_data(start=-2, end=-1, stride=2):
    dB = range(start, end, stride)
    smoothingLen = 11
    chL = 10
    img_rows, img_cols = 2, 12
    for db in dB:
        print("situation is: {}dB".format(db))
        begin_db = time.time()
        X_train, Y_train, X_test, Y_test = generateData(10000000, 9000000, db, smoothingLen, chL, 'cnn')
        X_train_crnn = np.asarray(X_train)
        Y_train_crnn = np.asarray(Y_train)
        X_test_crnn = np.asarray(X_test)
        Y_test_crnn = np.asarray(Y_test)
        # print("situation is: {}dB".format(db))
        # print("X_train_crnn is: ", X_train_crnn.shape)
        # print("Y_train_crnn is: ", Y_train_crnn.shape)
        # print("X_test_crnn is: ", X_test_crnn.shape)
        # print("Y_test_crnn is: ", Y_test_crnn.shape)

        X_train_crnn = X_train_crnn.reshape(X_train_crnn.shape[0], 1, img_rows, img_cols)
        X_test_crnn = X_test_crnn.reshape(X_test_crnn.shape[0], 1, img_rows, img_cols)
        # print("==========after shape=========")
        # print("X_train_crnn is: ", X_train_crnn.shape)
        # print("Y_train_crnn is: ", Y_train_crnn.shape)
        # print("X_test_crnn is: ", X_test_crnn.shape)
        # print("Y_test_crnn is: ", Y_test_crnn.shape)

        make_dir("./database_qpsk/")
        np.save("./database_qpsk/{}_train_x.npy".format(db), X_train_crnn) #读取文件
        np.save("./database_qpsk/{}_train_y.npy".format(db), Y_train_crnn) #读取文件
        np.save("./database_qpsk/{}_test_x.npy".format(db), X_test_crnn) #读取文件
        np.save("./database_qpsk/{}_test_y.npy".format(db), Y_test_crnn) #读取文件
        print("Data Saved in {}dB".format(db))
        del X_train, Y_train, X_test, Y_test
        end_db = time.time()
        print(db, 'db time', (end_db-begin_db)/60, 'min')


if __name__ == "__main__":
    begin = time.time()
    gen_save_data(start=0, end=51, stride=2)
    end = time.time()
    print('total time:', (end-begin)/60, 'min')

