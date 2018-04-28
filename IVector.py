#!/usr/bin/env python
# -*- coding:utf-8 -*-
#################################################################################
# In terms of paper 25:Deep Speaker Embeddings for Short-Duration Speaker Verification,
# NIST evaluations (2004-08) and a portion of the Switchboard dataset for training both
# deep networks and baseline i-vector/PLDA systems.
#
# Only use NIST-SRE 2010 female test setto test perfomance,but i will use all data available from 2010
# to be the test set
################################################################################
import os

os.environ["SIDEKIT"] = "theano=false,theano_config=cpu,libsvm=false,mpi=false"
from sidekit.features_extractor import FeaturesExtractor
from sidekit.features_server import FeaturesServer
import sidekit
from sidekit import Key
import pandas as pd
from PrepareData import IdMapConstructor as idc
import numpy as np
import IVector_Base
import jyh.Utils as ut
import h5py
import globalVar as glb

root = glb.get_root()
ivectorWorkDir = root + 'fea/ivector/'


class IV(IVector_Base.IVector_Base):
    @ut.timing("InitTime")
    def __init__(self, reGenOther, reGenKey):
        # parameters:################################################################
        self.distrib_nb = 2048  # number of Gaussian distributions for each GMM
        self.rank_TV = 400  # Rank of the total variability matrix
        self.tv_iteration = 10  # number of iterations to run
        self.plda_rk = 400  # rank of the PLDA eigenvalues matrix
        self.feature_dir = root + 'fea/fea/'  # directory where to find the features
        self.nbThread = 32  # Number of parallel process to run,cause machine has 32 cpus,i set it is 32
        #############################################################################

        # <editor-fold desc="idmap construct">
        # the list need to init:######################################################
        # • the list of files to train the GMM-UBM
        # • an IdMap listing the files to train the total variability matrix
        # • an IdMap to train the PLDA, WCCN, Mahalanobis matrices
        # • the IdMap listing the enrolment segments and models
        # • the IdMap describing the test segments
        #
        # the way of data using study by Paper 33:X-VECTORS: ROBUST DNN EMBEDDINGS FOR SPEAKER RECOGNITION,
        # use all train set to train GMM-UBM and TV,use SRE04-08 to train PLDA
        if reGenOther:
            idmapAll = idc.readAllSreIdmap()
            trial_key = idc.keyConstructor(reGenKey)
            self.enroll_idmap = sidekit.IdMap()
            self.enroll_idmap.leftids = trial_key.modelid
            self.enroll_idmap.rightids = np.array(['sre10/' + i[:-7] + i[-3:] for i in list(trial_key.modelset)])
            self.enroll_idmap.start = np.empty(np.size(self.enroll_idmap.leftids), '|O')
            self.enroll_idmap.stop = np.empty(np.size(self.enroll_idmap.leftids), '|O')
            self.test_idmap = sidekit.IdMap()
            self.test_idmap.leftids = trial_key.segid
            self.test_idmap.rightids = np.array(['sre10/' + i[:-7] + i[-3:] for i in list(trial_key.segset)])
            self.test_idmap.start = np.empty(np.size(self.test_idmap.leftids), '|O')
            self.test_idmap.stop = np.empty(np.size(self.test_idmap.leftids), '|O')
            self.test_ndx = idc.key2ndx_SelectSomeTrialToDo(trial_key)

            self.ubm_TV_idmap = sidekit.IdMap()
            self.plda_idmap = sidekit.IdMap()
            tl, tr, ul, ur = [], [], [], []
            for i, j in idmapAll.items():
                if i == '10':
                    continue
                if len(j) == 4:
                    tl += list(j[0].leftids) + list(j[1].leftids) + list(j[2].leftids) + list(j[3].leftids)
                    tr += ['sre0' + i + '/' + k[:-6] + k[-2:] for k in
                           list(j[0].rightids) + list(j[1].rightids) + list(j[2].rightids) + list(j[3].rightids)]
                else:
                    ul += list(j[0].leftids) + list(j[1].leftids)
                    ur += [i + '/' + k[:-6] + k[-2:] for k in list(j[0].rightids) + list(j[1].rightids)]
            self.ubm_TV_idmap.leftids = np.array(tl + ul)
            self.ubm_TV_idmap.rightids = np.array(tr + ur)
            self.ubm_TV_idmap.start = np.empty(np.size(self.ubm_TV_idmap.leftids), '|O')
            self.ubm_TV_idmap.stop = np.empty(np.size(self.ubm_TV_idmap.leftids), '|O')
            self.plda_idmap.leftids = np.array(tl)
            self.plda_idmap.rightids = np.array(tr)
            self.plda_idmap.start = np.empty(np.size(self.plda_idmap.leftids), '|O')
            self.plda_idmap.stop = np.empty(np.size(self.plda_idmap.leftids), '|O')
            self.trial_key = trial_key
            self.trial_key.write(root + "fea/trial_key.h5")
            self.test_ndx.write(root + "fea/test_ndx.h5")
            self.enroll_idmap.write(root + "fea/enroll_idmap.h5")
            self.test_idmap.write(root + "fea/test_idmap.h5")
            self.ubm_TV_idmap.write(root + "fea/ubm_TV_idmap.h5")
            self.plda_idmap.write(root + "fea/plda_idmap.h5")
        else:
            self.trial_key = sidekit.Key(root + "fea/trial_key.h5")
            self.test_ndx = sidekit.Ndx(root + "fea/test_ndx.h5")
            self.enroll_idmap = sidekit.IdMap(root + "fea/enroll_idmap.h5")
            self.test_idmap = sidekit.IdMap(root + "fea/test_idmap.h5")
            self.ubm_TV_idmap = sidekit.IdMap(root + "fea/ubm_TV_idmap.h5")
            self.plda_idmap = sidekit.IdMap(root + "fea/plda_idmap.h5")
        ###############################################################################
        # </editor-fold>

        # feature Server ###############################################################
        self.feaServer = sidekit.FeaturesServer(feature_filename_structure=self.feature_dir + "{}.h5",
                                                dataset_list=["cep"])
        s = self.feaServer.load('sre04/xgvxTA', input_feature_filename=None, label=None, start=None, stop=None)
        pass

    '''Step1: Create the universal background model from all the training speaker data'''

    def trainUBM(self, data=None):
        self.ubm = sidekit.Mixture()
        ubm_list = list(self.ubm_TV_idmap.rightids)
        llk = self.ubm.EM_split(self.feaServer, ubm_list, self.distrib_nb, num_thread=self.nbThread, save_partial=True,
                                output_file_name=ivectorWorkDir + 'ubm')
        self.ubm.write(ivectorWorkDir + 'ubm_{}'.format(self.distrib_nb))
        pass

    '''Step2.1: Calculate the statistics from train data set needed for the iVector model.'''

    def calStat(self, data=None):
        # enroll_stat = sidekit.StatServer(self.enroll_idmap, self.ubm)
        # enroll_stat.accumulate_stat(ubm=self.ubm, feature_server=self.feaServer, seg_indices=range(enroll_stat.segset.shape[0]), num_thread = self.nbThread)
        # enroll_stat.write('data/stat_sre10_core-core_enroll_{}.h5'.format(self.distrib_nb))
        # test_stat = sidekit.StatServer(self.test_idmap, self.ubm)
        # test_stat.accumulate_stat(ubm=self.ubm, feature_server=self.feaServer, seg_indices=range(test_stat.segset.shape[0]), num_thread = self.nbThread)
        # test_stat.write('data/stat_sre10_core-core_test_{}.h5'.format(self.distrib_nb))
        # back_idmap = plda_all_idmap.merge(self.test_idmap)
        # back_stat = sidekit.StatServer(back_idmap, self.ubm)
        # back_stat.accumulate_stat(ubm=self.ubm, feature_server=self.feaServer, seg_indices=range(back_stat.segset.shape[0]), num_thread = self.nbThread)
        # back_stat.write('data/stat_back_{}.h5'.format(self.distrib_nb))
        pass

    '''Step2.2: Learn the total variability subspace from all the train speaker data.'''

    def learnTV(self, data=None):
        pass

    '''Step2.3:Now compute the development ivectors of train data set for each speaker and channel.  The result is size tvDim x nSpeakers x nChannels.'''

    def computeDevIV(self, data=None):
        pass

    '''Step3.1:do LDA on the development iVectors to find the dimensions that matter.'''

    def trainLDA(self, data):
        pass

    '''Step3.2: Now train a Gaussian PLDA model with development i-vectors'''

    def trainGPLDA(self, data=None):
        pass

    '''Step4: now we have the channel and LDA models. Let's compute ivector and do lda with enrollment and test,respectively'''

    def computeIVAndDoLDA_OnEnrollOrTest(self, data=None):
        pass

    '''Step5: Now score the models with all the test data.'''

    def scoreAndVisualization(self, data=None):
        pass


def main():
    iv = IV(False, False)


if __name__ == '__main__':
    main()
