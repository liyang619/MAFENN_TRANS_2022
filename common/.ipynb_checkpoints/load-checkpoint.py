import numpy as np
import re

re_complex = re.compile(r'([+-]?[0-9\.]+)([+-][0-9\.]+)i')


def loadTx():
    return np.loadtxt('../datasets/data_Tx.txt')


def loadY():
    tmp = np.loadtxt('../datasets/data_y.txt')
    #return tmp
    def helper(arr):
        res = []
        for i in range(len(arr)):
            if i % 2 == 0:
                res.append(complex(arr[i], arr[i+1]))
        return res
    return map(helper, tmp)


def loadLabel():
    return np.loadtxt('../datasets/data_test_label.txt')


def loadTest():
    res = []
    with open('../datasets/data_test.txt') as f:
        arr = f.readlines()
        for a in arr:
            a = a.strip().split('\t')
            res.append(map(lambda x : np.complex(x[:-1]+'j') ,a))
    return res


def parseComplex(s):
    return re_complex.match(s).groups()

if __name__ == '__main__':
    # loadTx()
    # loadY()
    # loadLabel()
    print np.array(loadTest()).shape
    #print helper([1,2,3,4])
    #pass
