#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import os

os.environ["SIDEKIT"] = "theano=false,theano_config=cpu,libsvm=false,mpi=false"
import multiprocessing
import h5py
import re
import random
import pandas as pd
import jyh.Utils as ut
import numpy as np
import globalVar as glb
import sidekit
import pickle

root = glb.get_root()  # '/home/jyh/D/jyh/data/'
experimentsIdentify = 'ivector_all'
inputDir = root + 'fea/'  # fea and idmap input dir


class FeaServer(sidekit.FeaturesServer):
    def __init__(self, feature_filename_structure,
                 dataset_list=["fb"],
                 mask=None,
                 feat_norm=None,
                 keep_all_features=True,  # feature has onle vad rest part,so needn't this parameter
                 delta=False,
                 double_delta=False,
                 rasta=False,  # feature has do the post handle,so needn't do this again
                 context=None):
        super(FeaServer, self).__init__(feature_filename_structure=feature_filename_structure,
                                        dataset_list=dataset_list,
                                        mask=mask,
                                        feat_norm=feat_norm,
                                        keep_all_features=keep_all_features,
                                        delta=delta,
                                        double_delta=double_delta,
                                        rasta=rasta,
                                        context=context)

    def get_features(self, show, channel=0, input_feature_filename=None, label=None, start=None, stop=None):
        if input_feature_filename is not None:
            self.feature_filename_structure = input_feature_filename

        h5f = h5py.File(self.feature_filename_structure.format(show), "r")
        if show[:5] == 'sre10':
            show = re.findall('sre10/(.*)[T|M][A|B]', show)[0]
        else:
            show = re.findall('/([^/]{4,})[T|M][A|B]', show)[0]

        fbank = h5f["/".join((show, "fb"))].value
        # label = np.ones(fbank.shape[0], dtype='bool')
        h5f.close()
        return fbank  # , label


class DataInteger():
    '''
        multiReadProc1 is used to read all data to one h5 file,
        but multiReadProc is used to get infomation about data.
    '''

    @staticmethod
    def multiReadProc(shows, num_thread, feature_filename_structure):
        manager = multiprocessing.Manager()

        # f_=manager.list([f])
        lens = len(shows)
        feaM_ = manager.list([0] * lens)
        feaF_ = manager.list([0] * lens)
        label_ = manager.list([0] * lens)
        lock_ = manager.Lock()
        cou = manager.list([0])
        feature_filename_structure = list([feature_filename_structure]) * lens
        pool = multiprocessing.Pool(num_thread, initializer=DataInteger.globalVarinit,
                                    initargs=(lock_, feaM_, feaF_, label_, cou))
        pool.map(DataInteger.proc, zip(shows, list(range(lens)), feature_filename_structure))
        pool.close()
        pool.join()

        return np.array([list(label_), list(feaM_)])

    @staticmethod
    def globalVarinit(_lock, _feaM, _feaF, _label, co):
        global feaM_
        global label_
        global lock_
        global cou
        global feaF_
        cou = co
        label_ = _label
        feaM_ = _feaM
        feaF_ = _feaF
        lock_ = _lock

    @staticmethod
    def proc(show_):
        sho = show_[0]
        ind = show_[1]
        feature_filename_structure_ = show_[2]

        h5f = h5py.File(feature_filename_structure_.format(sho), "r")
        show1_ = sho
        # show1_ = re.sub("/", "_", sho)
        if sho[:5] == 'sre10':
            sho = re.findall('sre10/(.*)[T|M][A|B]', sho)[0]
            # show1_="sre10_"+re.findall('.*_.*_.*_(.*)',show1_)[0]
        else:
            sho = re.findall('/([^/]{4,})[T|M][A|B]', sho)[0]

        featM = h5f["/".join((sho, "cep"))].value
        # featF = h5f["/".join((sho, "fb"))].value
        h5f.close()

        with lock_:
            # feaM_[ind] = featM.astype('float32')
            # feaF_[ind] = featF.astype('float32')
            feaM_[ind] = featM.shape[0]
            label_[ind] = show1_
            cou[0] += 1
            print(cou[0])

    @staticmethod
    def proc1(sho, ind, feature_filename_structure_):
        '''
        apply_async cannot use in class
        :param show_:
        :return:
        '''
        # sho = show_[0]
        # ind = show_[1]
        # feature_filename_structure_ = show_[2]
        name = sho
        h5f = h5py.File(feature_filename_structure_.format(sho), "r")
        if sho[:5] == 'sre10':
            sho = re.findall('sre10/(.*)[T|M][A|B]', sho)[0]
            # show1_="sre10_"+re.findall('.*_.*_.*_(.*)',show1_)[0]
        else:
            sho = re.findall('/([^/]{4,})[T|M][A|B]', sho)[0]
        # sho = re.findall('/([^/]{4,})[T|M][A|B]', sho)[0]
        featM = h5f["/".join((sho, "fb"))].value
        h5f.close()

        with lock_:
            data_[name] = featM
            if ind % 200 == 0:
                print(ind)

    @staticmethod
    def multiReadProc1(shows, num_thread, feature_filename_structure, outputDir):

        h5f = h5py.File(outputDir, "w")
        if len(shows) > 30000:  # this is train set,each epoch handle only 10000 data
            with ut.Timing(outputDir[-8:-3]):
                for i in range(0, len(shows), 10000):
                    data_ = DataInteger.multiReadProc1_core(feature_filename_structure, num_thread, shows[i:i + 10000])
                    for i, j in data_.items():
                        h5f[i] = j
        else:
            with ut.Timing(outputDir[-8:-3]):
                data_ = DataInteger.multiReadProc1_core(feature_filename_structure, num_thread, shows)
                for i, j in data_.items():
                    h5f[i] = j
        h5f.close()

    @staticmethod
    def multiReadProc1_core(feature_filename_structure, num_thread, shows):
        manager = multiprocessing.Manager()
        # lens = len(shows)
        lock_ = manager.Lock()
        data_ = manager.dict()
        pool = multiprocessing.Pool(num_thread, initializer=DataInteger.globalVarinit1, initargs=(lock_, data_))
        for j, i in enumerate(shows):
            pool.apply_async(DataInteger.proc1, (i, j, feature_filename_structure))
        # pool.map(DataInteger.proc1, zip(shows, list(range(lens)), feature_filename_structure))
        pool.close()
        pool.join()
        data = dict(data_)
        return data

    @staticmethod
    def globalVarinit1(_lock, _data):
        global data_
        data_ = _data
        global lock_
        lock_ = _lock

    @ut.timing("DataInteger")
    def __init__(self, train=True):
        # self.getSummary()
        self.saveH5()
        # self.summaryVisulization()

    def summaryVisulization(self):
        import matplotlib.pyplot as plt
        import csv
        train = np.load(inputDir + "fea/frameInfoSummary_train.npy")
        enroll = np.load(inputDir + "fea/frameInfoSummary_enroll.npy")
        test = np.load(inputDir + "fea/frameInfoSummary_test.npy")
        with open(inputDir + "fea/frameInfoSummary_train.txt", "w+") as my_csv:  # writing the file as my_csv
            csvWriter = csv.writer(my_csv, delimiter=',', )  # using the csv module to write the file
            csvWriter.writerows(train)  # write every row in the matrix
        with open(inputDir + "fea/frameInfoSummary_enroll.txt", "w+") as my_csv:  # writing the file as my_csv
            csvWriter = csv.writer(my_csv, delimiter=',')  # using the csv module to write the file
            csvWriter.writerows(enroll)  # write every row in the matrix
        with open(inputDir + "fea/frameInfoSummary_test.txt", "w+") as my_csv:  # writing the file as my_csv
            csvWriter = csv.writer(my_csv, delimiter=',')  # using the csv module to write the file
            csvWriter.writerows(test)  # write every row in the matrix

            # np.savetxt(inputDir + "fea/frameInfoSummary_train.txt",train[1,:])
            # np.savetxt(inputDir + "fea/frameInfoSummary_enroll.txt", enroll[1,:])
            # np.savetxt(inputDir + "fea/frameInfoSummary_test.txt", test[1,:])

            # fig, (ax0, ax1,ax2) = plt.subplots(nrows=3, figsize=(9, 6))
            # # 第二个参数是柱子宽一些还是窄一些，越大越窄越密
            # ax0.hist(train[1,:], 40, normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
            # ax0.set_title("train")
            # # ax1.hist(enroll[1, :], 40, normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
            # # ax1.set_title("enroll")
            # # ax2.hist(test[1, :], 40, normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
            # # ax2.set_title("test")
            # fig.subplots_adjust(hspace=0.4)
            # plt.show()
            # print("finish")

    def getSummary(self):
        feature_filename_structure = root + "fea/fea/{}.h5"
        ubm_TV_idmap = sidekit.IdMap(root + "fea/ubm_TV_idmap.h5")
        ubm_list = list(ubm_TV_idmap.rightids)
        tmp = DataInteger.multiReadProc(ubm_list, 12, feature_filename_structure)
        np.save(inputDir + "fea/frameInfoSummary_train", tmp)
        enroll_idmap_10s = sidekit.IdMap(root + "fea/enroll_idmap_10s.h5")
        test_idmap_10s = sidekit.IdMap(root + "fea/test_idmap_10s.h5")
        enro, test = list(enroll_idmap_10s.rightids), list(test_idmap_10s.rightids)
        tmp = DataInteger.multiReadProc(enro, 48, feature_filename_structure)
        np.save(inputDir + "fea/frameInfoSummary_enroll", tmp)
        tmp = DataInteger.multiReadProc(test, 48, feature_filename_structure)
        np.save(inputDir + "fea/frameInfoSummary_test", tmp)
        print("finish...")

    def saveH5(self):
        outDir = root + "fea/fea/h5/"
        feature_filename_structure = root + "fea/fea/{}.h5"
        ubm_TV_idmap = sidekit.IdMap(root + "fea/ubm_TV_idmap.h5")
        ubm_list = list(ubm_TV_idmap.rightids)
        tmp = DataInteger.multiReadProc1(ubm_list, 12, feature_filename_structure, outDir + "train.h5")
        enroll_idmap_10s = sidekit.IdMap(root + "fea/enroll_idmap_10s.h5")
        test_idmap_10s = sidekit.IdMap(root + "fea/test_idmap_10s.h5")
        enro, test = list(enroll_idmap_10s.rightids), list(test_idmap_10s.rightids)
        tmp = DataInteger.multiReadProc1(enro, 12, feature_filename_structure, outDir + "enroll.h5")
        tmp = DataInteger.multiReadProc1(test, 12, feature_filename_structure, outDir + "test.h5")
        print("finish...")

    @staticmethod
    def readData():
        ubm_TV_idmap = sidekit.IdMap(root + "fea/ubm_TV_idmap.h5")
        ubm_list = list(ubm_TV_idmap.rightids)
        with h5py.File(root + "fea/fea/train_mfcc.h5", 'r') as f:
            for i in range(len(ubm_list)):
                s = f[re.sub("/", "_", ubm_list[i])].value
                pass


class Datainteger2():
    '''
    this class is used to generate data which is prepared for deep learning

    1.if use this dataset,i should copy this class to be Dataset, which is used in dataloader,or
    just copy 3 method:__init__ __getitem__ __len__
    (__getitem__ __len__ has been commonded for that is not needed in generating)

    2.then i should copy dataset file and relative *.pk file to the work directory
    '''

    # if just use data in

    @ut.timing("__init__")
    def __init__(self, idmapDir=inputDir + "dl_train_idmap.pk", utterNum=100, validUtterNum=2, frameRange=[200, 400],
                 frameInfo=inputDir + "fea/frameInfoSummary_train.npy",  # both valid and train info are in this file
                 indDir=inputDir + "dl_train_ind.pk", batch=64,
                 newFeaDir=inputDir + "fea/dl/data", oriFeaDir=inputDir + "fea/", numWorker=20):
        '''
        both train and valid data will all be stored in the same file,the first utterNum data is train
        and the rest validUtterNum data is valid

        :param idmapDir: use to get info to generate data and info
        :param utterNum: utterance number per speaker in train set
        :param validUtterNum: utterance number per speaker in valid set
        :param frameRange: the ranges of randomly generate frames in each batch
        :param frameInfo:
        :param indDir: save index and other info which is stored to disk
        :param batch: batchSize
        :param newFeaDir:new feature directory
        :param oriFeaDir:origin feature directory,this is not needn't in deep learning
        '''
        self.batchSize = batch
        self.utterNum = utterNum
        self.frameRange = frameRange
        self.validUtterNum = validUtterNum
        self.personNum, self.idmapDict, self.ind, self.modelid2key = self.prepareInfo(frameInfo, idmapDir, indDir)
        ###################################################################
        self.genData(self.ind, self.batchSize, oriFeaDir, newFeaDir, numWorker)
        ###################################################################
        # this code will be used in deep learning,and past line is used in generating data
        # self.hf = [h5py.File(inputDir + "fea/dl/{}{}.h5".format(style, i), "r") for i in range(self.batchSize)]
        ###################################################################
    # def __getitem__(self, index):
    #     '''
    #     self.ind: modelid,show,length,start,stop,k-----k represent data is in the k-th file
    #     '''
    #     ind = self.ind[index]
    #     data, label = [0] * len(ind), [0] * len(ind)
    #     for j, i in enumerate(ind):
    #         # data[j] = self.cmvnOrOtherTransform(self.fea[i[1]][i[3]:i[4], :40])[None, None]
    #         data[j] = self.cmvnOrOtherTransform(self.hf[i[5]]["{}_{}_{}".format(re.sub("/","_",i[1]),i[3],i[4])])[None, None]
    #         label[j] = self.modelid2key[i[0]]
    #         #:40 represent that only use fisrt 40 dim, e.g. there is not use delta and accelerate data
    #     return tc.from_numpy(np.concatenate(data)).type(tc.FloatTensor), \
    #            tc.from_numpy(np.array(label)).type(tc.LongTensor)
    #
    # def __len__(self):
    #     return len(self.ind)

    def prepareInfo(self, frameInfo, idmapDir, indDir):
        if not os.path.exists(idmapDir):  # the sta in idmap is length of frame
            idmapDict = self.selectData(idmapDir, frameInfo)
        else:
            idmapDict = pickle.load(open(idmapDir, "rb"))

        utterNum, validUtterNum, batchSize, frameRange = idmapDict.pop("info")
        modelid2key = idmapDict.pop("key")
        if self.batchSize != batchSize or self.utterNum != utterNum or self.validUtterNum != validUtterNum or \
                        self.frameRange[0] != frameRange[0] or self.frameRange[1] != frameRange[1]:
            '''must be re-generate,because parameter has been changed'''
            idmapDict = self.selectData(idmapDir, frameInfo)
            idmapDict.pop("info")
            modelid2key = idmapDict.pop("key")
            personNum = len(idmapDict)
            ind = self.genInd(indDir, idmapDict)
        else:
            '''needn't re-generate'''
            personNum = len(idmapDict)
            if not os.path.exists(indDir):
                ind = self.genInd(indDir, idmapDict)
            else:
                ind = pickle.load(open(indDir, "rb"))
        return personNum, idmapDict, ind, modelid2key

    def selectData(self, idmapDir, frameInfo):
        '''
        select data such that >=12000frames and much than 2 utterances for each person,
        this method is used singal for select data
        '''
        idmap = sidekit.IdMap(idmapDir[:-2] + "h5")

        train = np.load(frameInfo)
        trainList = [i for i in train.T if int(i[1]) >= 12000]  # >=12000
        idmap = idmap.filter_on_right([i[0] for i in trainList], True)
        trainList = dict(trainList)
        for j, i in enumerate(idmap.rightids):
            idmap.start[j] = trainList[i]
        idmapDict = self.idmap2Dict(idmap)
        idmapDict = {i: j for i, j in idmapDict.items() if j.shape[0] > 1}  # >1 utter
        modelid2Key = {i: j for j, i in enumerate(idmapDict.keys())}
        # idmapDict = pd.concat(idmapDict)
        # idmapDict = Dataset.dataFrame2Idmap(idmapDict)
        # add 1 term into idmapDict to store some info that will be used in __init__()
        idmapDict["info"] = [self.utterNum, self.validUtterNum, self.batchSize, self.frameRange]
        idmapDict["key"] = modelid2Key
        pickle.dump(idmapDict, open(idmapDir, "wb"))
        return idmapDict

    def genInd(self, indDir, idmapDict):
        ind = []
        for i, j in idmapDict.items():
            utterInd = np.random.randint(0, j.shape[0], self.utterNum + self.validUtterNum)
            for k in utterInd:
                ind.append([i, j.iloc[k, 1], int(j.iloc[k, 2]), None, None, None])  # modelid,show,length,None,None
        random.shuffle(ind)  # to randomly get batchSize data,i randomly shuffle list so that i can get data in order
        ind = [ind[i:i + self.batchSize] for i in range(0, len(ind), self.batchSize)]  # split data to all batches
        for i in range(len(ind)):
            frameLen = np.random.randint(self.frameRange[0], self.frameRange[1] + 1)  # frame's length in one batchSize
            for k, j in enumerate(range(len(ind[i]))):
                frameInd = np.random.randint(frameLen, ind[i][j][2])
                ind[i][j][3] = frameInd - frameLen
                ind[i][j][4] = frameInd  # modelid,show,length,start,stop
                ind[i][j][5] = k  # data is saved in the k-th file
        pickle.dump(ind, open(indDir, "wb"))
        return ind

    @staticmethod
    def idmap2Dict(idmap):
        '''
        :param idmap:
        :return: dict(pd.DataFrame),key is model id ,value is show of segments corresponds with model id
        '''
        idmapDf = Datainteger2.idMap2DataFrame(idmap)
        idmapDict = dict(list(idmapDf.groupby('id')))
        return idmapDict

    @staticmethod
    def idMap2DataFrame(idmap: sidekit.IdMap, onlyModelId: bool = False) -> pd.DataFrame:
        # onlyModelId is mean that if only get model id from year 10,so it should be false when yesr is not 10
        if onlyModelId:
            idmap.leftids = np.asarray([i[-5:] for i in idmap.leftids])
        return pd.DataFrame(np.asarray([idmap.leftids, idmap.rightids, idmap.start, idmap.stop]).T,
                            columns=['id', 'fi', 'sta', 'sto'])

    @staticmethod
    def init(lock_, data_):
        global data
        data = data_
        global lock
        lock = lock_

    @staticmethod
    def getData_core(param):
        j, ind, feature_filename_structure = param
        feaSer = FeaServer(feature_filename_structure=feature_filename_structure)
        tmp = feaSer.get_features(ind[1])
        rang = tmp.shape[0] - ind[2] + 1
        if rang <= 0:
            rps = int(np.ceil((-rang + 1) / tmp.shape[0]))
            tmp = np.tile(tmp, (rps + 1, 1))[0, ind[2]]
        with lock:
            data["{}_{}_{}".format(re.sub("/", "_", ind[1]), ind[3], ind[4])] = tmp[ind[3]:ind[4],
                                                                                :40]  # onle use first 40 dim

    @staticmethod
    @ut.timing("genData")
    def genData(ind, batchSize, oriFeaDir, newFeaDir, numWorker):
        '''
        this method is used to generate data ,should be runed in single.
        :param ind: [modelid,show,length,start,stop,k]
        :param batchSize:
        :return:
        '''
        # feaSer = [FeaServer(feature_filename_structure=oriFeaDir + "{}.h5")] * batchSize
        manager = multiprocessing.Manager()
        data = manager.dict()
        lock = manager.Lock()
        pool = multiprocessing.Pool(processes=numWorker, initializer=Datainteger2.init, initargs=(lock, data))
        indS = [list() for i in range(batchSize)]
        for i in ind:
            for l, k in enumerate(i):
                indS[l].append(k)
        for j, i in enumerate(indS):
            pool.map(Datainteger2.getData_core, zip(range(len(i)), i, [oriFeaDir + "{}.h5"] * len(i)))
            hf = h5py.File(newFeaDir + "{}.h5".format(j), "w")
            for l, k in data.items():
                hf[l] = k
            hf.close()
            data.clear()
        pool.close()
        pool.join()


if __name__ == '__main__':
    a = Datainteger2()
