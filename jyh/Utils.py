#!/usr/bin/env python
# -*- coding:utf-8 -*-
import functools
import time
from multiprocessing import Pool, Manager
import struct
import numpy as np
import os
import pandas as pd
import re
# from graphviz import Digraph
# import h5py
import scipy.io as io
import pickle as pk

EX1 = "use decorator @log that you should append a string parameter in your return tuple so we can write it to log file\n"

logPath = "./log.txt"


class HTKFile:
    """ Class to load binary HTK file.

        Details on the format can be found online in HTK Book chapter 5.7.1.

        Not everything is implemented 100%, but most features should be supported.

        Not implemented:
            CRC checking - files can have CRC, but it won'tc be checked for correctness

            VQ - Vector features are not implemented.
    """

    data = None
    nSamples = 0
    nFeatures = 0
    sampPeriod = 0
    basicKind = None
    qualifiers = None

    # easy to call this class
    def __call__(self, filename):
        return self.load(filename)

    def load(self, filename):
        """ Loads HTK file.

            After loading the file you can check the following members:

                data (matrix) - data contained in the file

                nSamples (int) - number of frames in the file

                nFeatures (int) - number if features per frame

                sampPeriod (int) - sample period in 100ns units (e.g. fs=16 kHz -> 625)

                basicKind (string) - basic feature kind saved in the file

                qualifiers (string) - feature options present in the file

        """
        with open(filename, "rb") as f:

            header = f.read(12)
            self.nSamples, self.sampPeriod, sampSize, paramKind = struct.unpack(">iihh", header)
            basicParameter = paramKind & 0x3F

            if basicParameter is 0:
                self.basicKind = "WAVEFORM"
            elif basicParameter is 1:
                self.basicKind = "LPC"
            elif basicParameter is 2:
                self.basicKind = "LPREFC"
            elif basicParameter is 3:
                self.basicKind = "LPCEPSTRA"
            elif basicParameter is 4:
                self.basicKind = "LPDELCEP"
            elif basicParameter is 5:
                self.basicKind = "IREFC"
            elif basicParameter is 6:
                self.basicKind = "MFCC"
            elif basicParameter is 7:
                self.basicKind = "FBANK"
            elif basicParameter is 8:
                self.basicKind = "MELSPEC"
            elif basicParameter is 9:
                self.basicKind = "USER"
            elif basicParameter is 10:
                self.basicKind = "DISCRETE"
            elif basicParameter is 11:
                self.basicKind = "PLP"
            else:
                self.basicKind = "ERROR"

            self.qualifiers = []
            if (paramKind & 0o100) != 0:
                self.qualifiers.append("E")
            if (paramKind & 0o200) != 0:
                self.qualifiers.append("N")
            if (paramKind & 0o400) != 0:
                self.qualifiers.append("D")
            if (paramKind & 0o1000) != 0:
                self.qualifiers.append("A")
            if (paramKind & 0o2000) != 0:
                self.qualifiers.append("C")
            if (paramKind & 0o4000) != 0:
                self.qualifiers.append("Z")
            if (paramKind & 0o10000) != 0:
                self.qualifiers.append("K")
            if (paramKind & 0o20000) != 0:
                self.qualifiers.append("0")
            if (paramKind & 0o40000) != 0:
                self.qualifiers.append("V")
            if (paramKind & 0o100000) != 0:
                self.qualifiers.append("T")

            if "C" in self.qualifiers or "V" in self.qualifiers or self.basicKind is "IREFC" or self.basicKind is "WAVEFORM":
                self.nFeatures = sampSize // 2
            else:
                self.nFeatures = sampSize // 4

            if "C" in self.qualifiers:
                self.nSamples -= 4

            if "V" in self.qualifiers:
                raise NotImplementedError("VQ is not implemented")

            self.data = []
            if self.basicKind is "IREFC" or self.basicKind is "WAVEFORM":
                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        val = struct.unpack_from(">h", s, v * 2)[0] / 32767.0
                        frame.append(val)
                    self.data.append(np.array(frame))
            elif "C" in self.qualifiers:

                A = []
                s = f.read(self.nFeatures * 4)
                for x in range(self.nFeatures):
                    A.append(struct.unpack_from(">F", s, x * 4)[0])
                B = []
                s = f.read(self.nFeatures * 4)
                for x in range(self.nFeatures):
                    B.append(struct.unpack_from(">F", s, x * 4)[0])

                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        frame.append((struct.unpack_from(">h", s, v * 2)[0] + B[v]) / A[v])
                    self.data.append(np.array(frame))
            else:
                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        val = struct.unpack_from(">F", s, v * 4)
                        frame.append(val[0])
                    self.data.append(np.array(frame))
            self.data = np.array(self.data)

            if "K" in self.qualifiers:
                print("CRC checking not implememnted...")
        return self


def p():
    def log_decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kw):
            count = 1
            rlt = f(*args, **kw)
            print('[%d]\n' % (count))
            count += 1
            return rlt

        return wrapper

    return log_decorator


def getFileList(fileP):
    file = os.listdir(fileP)
    return [fileP + i for i in file]


#  compute the cose of time
def timing(type):
    def log_decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kw):
            start = time.time()
            rlt = f(*args, **kw)
            elapse = time.time() - start
            print('[%s]  time: %.2f s' % (type, elapse))
            return rlt

        return wrapper

    return log_decorator


class Timing(object):
    '''
    用上下文管理器计时
    e.g.:
    with MyTimer() as tc:
        test(1,2)
        time.sleep(1)
        print 'do other things'
    '''

    # __init is not must needed, just because i want to get a "type" parameter to show in the __exit__() method
    def __init__(self, type):
        self.type = type

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapse = time.time() - self.start
        print('[%s]  time: %.2f s' % (self.type, elapse))


def listDirRecursive(dir):
    t = []
    for fpathe, dirs, fs in os.walk(dir):
        for f in fs:
            t.append(os.path.join(fpathe, f))
    return t


# if filePath is None then the default direction is './log.txt'
def log(filePath):
    def log_decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kw):
            filePath1 = createDir(filePath)
            rlt = f(*args, **kw)
            if rlt == None:
                raise Exception(EX1)
            if isinstance(rlt, tuple):  # if rlt is tuple ,the reture is much more 1 parameter,so we only use last one
                with open(filePath1, "a") as fileW:
                    fileW.writelines(str(rlt[-1]))
                    if len(rlt[:-1]) == 1:
                        rlt = rlt[0]
                    else:
                        rlt = rlt[:-1]
            else:
                with open(filePath1, "a") as fileW:
                    fileW.writelines(str(rlt))
                    rlt = None
            return rlt

        return wrapper

    return log_decorator


def createDir(filePath):
    filePath1 = filePath
    # filePath is external variable ,
    # cannot be revised in the method (if must revised,we can add trial_key word 'global' or assign another variable 'filePath1' to handle)
    if filePath1 == None:
        filePath1 = logPath
    file_dir = os.path.split(filePath1)[0]
    # 判断文件路径是否存在，如果不存在，则创建，此处是创建多级目录
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return filePath1


class Log(object):
    def __init__(self, logStr, filePath=None):
        self.filePath = filePath
        self.logStr = logStr

    def __enter__(self):
        # nothing to do
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        filePath1 = createDir(self.filePath)
        with open(filePath1, "a") as fileW:
            fileW.writelines(str(self.logStr))
            print(str(self.logStr))


def logger(logStr, filePath=None):
    filePath1 = createDir(filePath)
    with open(filePath1, "a") as fileW:
        fileW.writelines(str(logStr))
        print(str(logStr))


def proc(fname):
    tmp = np.load(fname[0])
    size = np.shape(tmp)
    label = int(re.findall("(\d+)\.npy", fname[0])[0]) - 1
    # listData = []
    # with open(fname) as f:
    #     lines = f.readlines()
    #     for line in lines:  # 先把逐行数据取出来
    #         line_data = line.strip('\n').split(' ')
    #         line_data = [float(i) for i in line_data if i != '']
    #         listData.append(line_data)  # 再通过处理，放回到list列表中
    # tmp = np.array(listData)
    # size = np.shape(tmp)
    # label = int(re.findall(".*\.(\d+)\.cut", fname)[0]) - 1

    with lock_mul:
        data_mul[fname[1]] = tmp
        dataSize_mul[fname[1]] = size
        label_mul[fname[1]] = label
        # data_mul.append(tmp)
        # dataSize_mul.append(size)
        # label_mul.append(label)


def multiReadProc(files):
    manager = Manager()
    data_mul = manager.list()
    dataSize_mul = manager.list()
    label_mul = manager.list()
    lock_mul = manager.Lock()
    lens = len(files)
    for i in range(lens):
        dataSize_mul.append(0)
        label_mul.append(0)
        data_mul.append(0)

    pool = Pool(initializer=globalVarinit,
                initargs=(lock_mul, data_mul, dataSize_mul, label_mul))  # default number of processes is os.cpu_count()
    pool.map(proc, zip(files, list(range(lens))))
    pool.close()
    pool.join()
    tmp1 = data_mul
    tmp2 = dataSize_mul
    tmp3 = label_mul
    return tmp1, tmp2, tmp3


def globalVarinit(_lock, _data, _dataSize, _label):
    global data_mul
    global dataSize_mul
    global label_mul
    global lock_mul
    data_mul = _data
    lock_mul = _lock
    dataSize_mul = _dataSize
    label_mul = _label


# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    child = []
    pathDir = os.listdir(os.path.expanduser(filepath))
    for allDir in pathDir:
        child.append(allDir)
    return sorted(child)
    # print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题


# fast algorithm to calculate the euclidean distances between each vector from matrix A and matrix B
def euclideanDistances(A, B):
    if not isinstance(A, np.matrix):
        A = np.matrix(A)
    if not isinstance(B, np.matrix):
        B = np.matrix(B)
    BT = B.transpose()
    vecProd = A * BT
    SqA = A.getA() ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B.getA() ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd

    ED = SqED.getA()
    ED[ED < 0] = 0  # some value close to zero but is negative,so do this step
    ED = np.nan_to_num(ED)  # may has some nan value,this method used to transform nan to zero
    ED = ED ** 0.5
    # ED=np.matrix(ED) # return matrix,if comment this code,return array.
    return ED


def showAsGridGraghs(size_figure_grid, data, show=False, savePath=None):  # data.shape is (batch,w,h,channel)
    # if data is torch,then we can do :
    # import torchvision.utils as vutils
    # vutils.save_image(data, 'path')
    import itertools
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(size_figure_grid, size_figure_grid))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid ** 2):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.transpose(data[k], (1, 2, 0)))  # , cmap='gray')

    if savePath:
        plt.savefig(savePath)

    if show:
        plt.show()
    else:
        plt.close()

# def genrateHDF5():
#     F = h5py.File("./1.hdf5", "w")
#     for i in ["train","test"]:
#         root="/home/jiangyiheng/matlabProjects/xiangmu/mfcc/"+i+"/"
#         files = os.listdir(root)
#         files.sort()
#         files = [root + i for i in files]
#
#         d, s, l = multiReadProc(files)
#         g1 = F.create_group(i)
#         g1.create_dataset("data", d)
#         g1.create_dataset("dataSize", s)
#         g1.create_dataset("label", l)
#
#
# if __name__ == '__main__':
#     genrateHDF5()
#     pass
