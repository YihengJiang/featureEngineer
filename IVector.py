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
import logging
import os
import re
import sys

from sidekit.mixture import sum_log_probabilities
from sidekit.sidekit_wrappers import process_parallel_lists
from sidekit.statserver import compute_llk, ct
os.environ["SIDEKIT"] = "theano=false,theano_config=cpu,libsvm=false,mpi=false"
# os.environ["MKL_THREADING_LAYER"]="GNU"
from sidekit.features_server import FeaturesServer
import sidekit
from sidekit import sidekit_io
import copy
from PrepareData import IdMapConstructor as idc
import numpy as np
import IVector_Base
import jyh.Utils as ut
import h5py
import globalVar as glb
import multiprocessing
import warnings
import ctypes

root = glb.get_root()
ivectorWorkDir = root + 'fea/ivector/'
logger = ut.logByLogginModule("IVector")
saveFlag = "10s"

STAT_TYPE = np.float64


class FeaServer(sidekit.FeaturesServer):
    def __init__(self, feature_filename_structure=None, dataset_list=None):
        super(FeaServer, self).__init__(feature_filename_structure=feature_filename_structure,
                                        dataset_list=dataset_list)
        self.fea = None  # dict(show:feature)

    # def stack_features_parallel(self,  # fileList, numThread=1):
    #                             show_list,
    #                             channel_list=None,
    #                             feature_filename_list=None,
    #                             label_list=None,
    #                             start_list=None,
    #                             stop_list=None,
    #                             num_thread=1):
    #     if channel_list is None:
    #         channel_list = np.zeros(len(show_list))
    #     if feature_filename_list is None:
    #         feature_filename_list = np.empty(len(show_list), dtype='|O')
    #     if label_list is None:
    #         label_list = np.empty(len(show_list), dtype='|O')
    #     if start_list is None:
    #         start_list = np.empty(len(show_list), dtype='|O')
    #     if stop_list is None:
    #         stop_list = np.empty(len(show_list), dtype='|O')
    #
    #     # queue_in = Queue.Queue(maxsize=len(fileList)+numThread)
    #     queue_in = multiprocessing.JoinableQueue(maxsize=len(show_list) + num_thread)
    #     queue_out = []
    #
    #     # Start worker processes
    #     jobs = []
    #     for i in range(num_thread):
    #         queue_out.append(multiprocessing.Queue())
    #         p = multiprocessing.Process(target=self._stack_features_worker,
    #                                     args=(queue_in, queue_out[i]))
    #         jobs.append(p)
    #         p.start()
    #
    #     # Submit tasks
    #     for task in zip(show_list, channel_list, feature_filename_list, label_list, start_list, stop_list):
    #         queue_in.put(task)
    #
    #     # Add None to the queue to kill the workers
    #     for task in range(num_thread):
    #         queue_in.put(None)
    #
    #     # Wait for all the tasks to finish
    #     queue_in.join()
    #
    #     output = []
    #     for q in queue_out:
    #         while True:
    #             data = q.get()
    #             if data is None:
    #                 break
    #             output.append(data)
    #
    #     for p in jobs:
    #         p.join()
    #     return output  # np.concatenate(output, axis=0)

    # def get_features(self, show, channel=0, input_feature_filename=None, label=None, start=None, stop=None):
    #     if input_feature_filename is not None:
    #         self.feature_filename_structure = input_feature_filename
    #
    #     h5f = h5py.File(self.feature_filename_structure.format(show), "r")
    #
    #     show_ = copy.deepcopy(show)
    #     if show[:5] == 'sre10':
    #         show = re.findall('sre10/(.*)[T|M][A|B]', show)[0]
    #     else:
    #         show = re.findall('/([^/]{4,})[T|M][A|B]', show)[0]
    #     # Get the selected segment
    #     dataset_length = h5f[show + "/" + next(h5f[show].__iter__())].shape[0]
    #     # Deal with the case where start < 0 or stop > feat.shape[0]
    #     start = 0
    #     stop = dataset_length
    #
    #     feat = []
    #     if "energy" in self.dataset_list:
    #         feat.append(h5f["/".join((show, "energy"))].value[start:stop, np.newaxis])
    #     if "cep" in self.dataset_list:
    #         feat.append(h5f["/".join((show, "cep"))][start:stop, :])
    #     if "fb" in self.dataset_list:
    #         feat.append(h5f["/".join((show, "fb"))][start:stop, :])
    #     feat = np.hstack(feat)
    #
    #     h5f.close()
    #
    #     return [[show_, feat]]

    def stack_features_parallel2(self, show_list, num_thread=1):
        featuresL, labelL = FeaServer.multiReadProc(show_list, num_thread, self.feature_filename_structure,
                                                    self.dataset_list)
        return dict(zip(labelL, featuresL))

    @staticmethod
    def multiReadProc(shows, num_thread, feature_filename_structure, dataset_list):
        lens = len(shows)
        manager = multiprocessing.Manager()
        fea_ = manager.list([0] * lens)
        label_ = manager.list([0] * lens)
        lock_ = manager.Lock()

        pool = multiprocessing.Pool(num_thread, initializer=FeaServer.globalVarinit,
                                    initargs=(lock_, fea_, label_, feature_filename_structure, dataset_list))
        pool.map(FeaServer.proc, zip(shows, list(range(lens))))
        pool.close()
        pool.join()
        return fea_, label_

    @staticmethod
    def globalVarinit(_lock, _fea, _label, _feature_filename_structure, _dataset_list):
        global fea_
        global label_
        global lock_
        global feature_filename_structure_
        global dataset_list_
        feature_filename_structure_ = _feature_filename_structure
        dataset_list_ = _dataset_list
        label_ = _label
        fea_ = _fea
        lock_ = _lock

    @staticmethod
    def proc(show_):
        sho = show_[0]
        ind = show_[1]

        h5f = h5py.File(feature_filename_structure_.format(sho), "r")

        show1_ = copy.deepcopy(sho)
        if sho[:5] == 'sre10':
            sho = re.findall('sre10/(.*)[T|M][A|B]', sho)[0]
        else:
            sho = re.findall('/([^/]{4,})[T|M][A|B]', sho)[0]

        feat = []
        if "energy" in dataset_list_:
            feat.append(h5f["/".join((sho, "energy"))].value[:, np.newaxis])
        if "cep" in dataset_list_:
            feat.append(h5f["/".join((sho, "cep"))])
        if "fb" in dataset_list_:
            feat.append(h5f["/".join((sho, "fb"))])
        feat = np.hstack(feat)
        h5f.close()
        with lock_:
            fea_[ind] = feat
            label_[ind] = show1_


class Mixture(sidekit.Mixture):
    def __init__(self):
        super(Mixture, self).__init__()

    def EM_split(self,
                 features_server,
                 feature_list,
                 distrib_nb,
                 iterations=(1, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8),
                 num_thread=1,
                 llk_gain=0.01,
                 save_partial=False,
                 output_file_name="ubm",
                 ceil_cov=10,
                 floor_cov=1e-2):
        llk = []
        with ut.Timing("ubm data extract"):
            self._init(features_server, feature_list, num_thread)
        # for N iterations:
        for it in iterations[:int(np.log2(distrib_nb))]:
            # Save current model before spliting
            if save_partial:
                self.write('{}_{}g.h5'.format(output_file_name, self.get_distrib_nb()), prefix='')

            self._split_ditribution()

            # initialize the accumulator
            accum = copy.deepcopy(self)

            for i in range(it):
                accum._reset()

                # serialize the accum
                accum._serialize()
                llk_acc = np.zeros(1)
                sh = llk_acc.shape
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    tmp = multiprocessing.Array(ctypes.c_double, llk_acc.size)
                    llk_acc = np.ctypeslib.as_array(tmp.get_obj())
                    llk_acc = llk_acc.reshape(sh)

                with ut.Timing('ubm Expectation and Maximization:' + str(i)):
                    # E step
                    self._expectation_list(stat_acc=accum,
                                           feature_list=feature_list,
                                           feature_server=features_server,
                                           llk_acc=llk_acc,
                                           num_thread=num_thread)
                    llk.append(llk_acc[0] / np.sum(accum.w))
                    # M step
                    self._maximization(accum, ceil_cov=ceil_cov, floor_cov=floor_cov)

                if i > 0:
                    gain = llk[-1] - llk[-2]

                    if gain < llk_gain:
                        logger.info('EM (break) distrib_nb: %d %i/%d gain: %f' % (
                            self.mu.shape[0], i + 1, it, gain))
                        break
                    else:
                        logger.info('EM (continu) distrib_nb: %d %i/%d gain: %f' % (
                            self.mu.shape[0], i + 1, it, gain))
                        continue
                    pass
                else:
                    logger.info('EM (start) distrib_nb: %d %i/%d gain: %f' % (
                        self.mu.shape[0], i + 1, it, llk[-1]))
                    pass

        return llk

    @process_parallel_lists
    def _expectation_list(self, stat_acc, feature_list, feature_server, llk_acc=np.zeros(1), num_thread=1):
        stat_acc._reset()
        for feat in feature_list:
            # cep = feature_server.load(feat)[0]
            cep = feature_server.fea[feat]
            llk_acc[0] += self._expectation(stat_acc, cep)

    def _init(self, features_server, feature_list, num_thread=1):
        features = features_server.stack_features_parallel(feature_list, num_thread=num_thread)
        #############save feature and fileId#############################
        fea = []
        for i in features:  # show,fea
            fea.append(i[1])
        features_server.fea = dict(features)
        #################################################################
        features = np.concatenate(fea, axis=0)
        mu = features.mean(0)
        cov = (features ** 2).mean(0)

        # n_frames, mu, cov = mean_std_many(features_server, feature_list, in_context=False, num_thread=num_thread)
        self.mu = mu[None]
        self.invcov = 1. / cov[None]
        self.w = np.asarray([1.0])
        self.cst = np.zeros(self.w.shape)
        self.det = np.zeros(self.w.shape)
        self.cov_var_ctl = 1.0 / copy.deepcopy(self.invcov)
        self._compute_all()


class StatServer(sidekit.StatServer):
    def __init__(self, statserver_file_name=None, distrib_nb=0, feature_size=0, index=None):
        # super(StatServer, self).__init__(statserver_file_name, distrib_nb, feature_size, index)
        self.modelset = np.empty(0, dtype="|O")
        self.segset = np.empty(0, dtype="|O")
        self.start = np.empty(0, dtype="|O")
        self.stop = np.empty(0, dtype="|O")
        self.stat0 = np.array([], dtype=STAT_TYPE)
        self.stat1 = np.array([], dtype=STAT_TYPE)

        if isinstance(statserver_file_name, str) and index is None:
            tmp = StatServer.read(statserver_file_name)
            self.modelset = tmp.modelset
            self.segset = tmp.segset
            self.start = tmp.start
            self.stop = tmp.stop
            self.stat0 = tmp.stat0
            self.stat1 = tmp.stat1

        elif isinstance(statserver_file_name, sidekit.IdMap):
            self.modelset = statserver_file_name.leftids
            self.segset = statserver_file_name.rightids
            self.start = statserver_file_name.start
            self.stop = statserver_file_name.stop
            self.stat0 = multiprocessing.Manager().list()
            self.stat1 = multiprocessing.Manager().list()

    def partition(self, part):
        self.modelset = self.modelset[:part]
        self.segset = self.segset[:part]
        self.start = self.start[:part]
        self.stop = self.stop[:part]
        self.stat0 = self.stat0[:part]
        self.stat1 = self.stat1[:part]

    @staticmethod
    def proc(show_idx):
        cep = server_.fea[show_idx[0]]
        if not ubm_.dim() == cep.shape[1]:
            raise Exception('dimension of ubm and features differ: {:d} / {:d}'.format(ubm_.dim(), cep.shape[1]))
        else:
            if ubm_.invcov.ndim == 2:
                lp = ubm_.compute_log_posterior_probabilities(cep)
            else:
                lp = ubm_.compute_log_posterior_probabilities_full(cep)
            pp, foo = sum_log_probabilities(lp)

            with lock_:
                # Compute 0th-order statistics
                stat0_[show_idx[1]] = pp.sum(0).astype(STAT_TYPE)
                # Compute 1st-order statistics
                stat1_[show_idx[1]] = np.reshape(np.transpose(
                    np.dot(cep.transpose(), pp)), ubm_.sv_size()).astype(STAT_TYPE)

    @staticmethod
    def multiReadProc(shows, num_thread, ubm, server):
        lens = len(shows)
        manager = multiprocessing.Manager()
        stat0_ = manager.list([0] * lens)
        stat1_ = manager.list([0] * lens)
        lock_ = manager.Lock()

        pool = multiprocessing.Pool(num_thread, initializer=StatServer.globalVarinit,
                                    initargs=(
                                        lock_, stat0_, stat1_, ubm,
                                        server))  # default number of processes is os.cpu_count()
        pool.map(StatServer.proc, zip(shows, list(range(lens))))
        pool.close()
        pool.join()
        return stat0_, stat1_

    @staticmethod
    def globalVarinit(_lock, _data, _data1, _ubm, _server):
        global stat0_
        global stat1_
        global lock_
        global ubm_
        global server_
        ubm_ = _ubm
        server_ = _server
        stat1_ = _data1
        stat0_ = _data
        lock_ = _lock

    # use my own code
    # @process_parallel_lists
    def accumulate_stat(self, ubm, feature_server, seg_indices=None, channel_extension=("", "_b"), num_thread=1):
        stat0, stat1 = StatServer.multiReadProc(list(self.segset), num_thread, ubm, feature_server)
        self.stat0 = np.array(stat0)
        self.stat1 = np.array(stat1)

    # use to TV estimate
    def estimate_between_class(self,
                               itNb,
                               V,
                               mean,
                               sigma_obs,
                               batch_size=100,
                               Ux=None,
                               Dz=None,
                               minDiv=True,
                               num_thread=1,
                               re_estimate_residual=False,
                               save_partial: str = None):

        # Initialize the covariance
        sigma = sigma_obs

        # Estimate F by iterating the EM algorithm
        for it in range(itNb):
            with ut.Timing("TV EM algorithm"):
                logger.info('Estimate between class covariance, it %d / %d',
                            it + 1, itNb)

                # Dans la fonction estimate_between_class
                model_shifted_stat = copy.deepcopy(self)

                # subtract channel effect, Ux, if already estimated
                if Ux is not None:
                    model_shifted_stat = model_shifted_stat.subtract_weighted_stat1(Ux)

                # Sum statistics per speaker
                model_shifted_stat, session_per_model = model_shifted_stat.sum_stat_per_model()
                # subtract residual, Dz, if already estimated
                if Dz is not None:
                    model_shifted_stat = model_shifted_stat.subtract(Dz)

                    # E-step
                _A, _C, _R = model_shifted_stat._expectation(V, mean, sigma, session_per_model, batch_size, num_thread)

                if not minDiv:
                    _R = None

                # M-step
                if re_estimate_residual:
                    V, sigma = model_shifted_stat._maximization(V, _A, _C, _R, sigma_obs, session_per_model.sum())
                else:
                    V = model_shifted_stat._maximization(V, _A, _C, _R)[0]

                if sigma.ndim == 2:
                    logger.info('Likelihood after iteration %d / %f', it + 1, compute_llk(self, V, sigma))

                del model_shifted_stat

                if save_partial:
                    sidekit.sidekit_io.write_fa_hdf5((mean, V, None, None, sigma),
                                                     save_partial + "_{}_between_class.h5".format(it))

        return V, sigma

    @staticmethod
    def read(statserver_file_name, prefix=''):
        """Read StatServer in hdf5 format

        :param statserver_file_name: name of the file to read from
        :param prefix: prefixe of the dataset to read from in HDF5 file
        """
        with h5py.File(statserver_file_name, "r") as f:
            statserver = StatServer()
            statserver.modelset = f.get(prefix + "modelset").value
            statserver.segset = f.get(prefix + "segset").value

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                statserver.modelset = statserver.modelset.astype('U', copy=False)
                statserver.segset = statserver.segset.astype('U', copy=False)

            tmpstart = f.get(prefix + "start").value
            tmpstop = f.get(prefix + "stop").value
            statserver.start = np.empty(f[prefix + "start"].shape, '|O')
            statserver.stop = np.empty(f[prefix + "stop"].shape, '|O')
            statserver.start[tmpstart != -1] = tmpstart[tmpstart != -1]
            statserver.stop[tmpstop != -1] = tmpstop[tmpstop != -1]

            statserver.stat0 = f.get(prefix + "stat0").value.astype(dtype=STAT_TYPE)
            statserver.stat1 = f.get(prefix + "stat1").value.astype(dtype=STAT_TYPE)

            assert statserver.validate(), "Error: wrong StatServer format"
            return statserver


class IV(IVector_Base.IVector_Base):
    @ut.timing("InitTime")
    def __init__(self):
        self.ubm = None
        self.back_stat = None
        self.enroll_stat = None
        self.test_stat = None
        self.Tmatrix = None
        self.train_iv, self.enroll_iv, self.test_iv = None, None, None
        self.train_iv_lda, self.enroll_iv_lda, self.test_iv_lda = None, None, None

        # parameters:################################################################
        self.distrib_nb = 2048  # number of Gaussian distributions for each GMM
        self.rank_TV = 400  # Rank of the total variability matrix
        self.tv_iteration = 5  # number of iterations to run
        self.plda_rk = 400  # rank of the PLDA eigenvalues matrix,if use lda or anyother operate,this value will change
        self.feature_dir = root + 'fea/fea/'  # directory where to find the features
        self.nbThread = 48  # Number of parallel process to run,cause machine has 32 cpus,i set it is 32
        self.feaSize = 72  # feature size that include (23mfcc+1energy)+24derived+24accelerate
        self.batchSize = 100  # this is use in  calculating stat and ivector,but plda's batch is much bigger than it(as it set to 1000)
        self.lda_rk = 200  # rank of lda
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

        # if not os.path.exists(root + "fea/trial_key_custom.h5"):
        #     idmapAll = idc.readAllSreIdmap()
        #     trial_key = idc.keyConstructorCustom(True)
        #     self.enroll_idmap = sidekit.IdMap()
        #     self.enroll_idmap.leftids = trial_key.modelid
        #     self.enroll_idmap.rightids = np.array(['sre10/' + i[:-7] + i[-3:] for i in list(trial_key.modelset)])
        #     self.enroll_idmap.start = np.empty(np.size(self.enroll_idmap.leftids), '|O')
        #     self.enroll_idmap.stop = np.empty(np.size(self.enroll_idmap.leftids), '|O')
        #     self.test_idmap = sidekit.IdMap()
        #     self.test_idmap.leftids = trial_key.segid
        #     self.test_idmap.rightids = np.array(['sre10/' + i[:-7] + i[-3:] for i in list(trial_key.segset)])
        #     self.test_idmap.start = np.empty(np.size(self.test_idmap.leftids), '|O')
        #     self.test_idmap.stop = np.empty(np.size(self.test_idmap.leftids), '|O')
        #     self.test_ndx = idc.key2ndx_SelectSomeTrialToDo(trial_key)
        #
        #     self.ubm_TV_idmap = sidekit.IdMap()
        #     self.plda_idmap = sidekit.IdMap()
        #     tl, tr, ul, ur = [], [], [], []
        #     for i, j in idmapAll.items():
        #         if i == '10':
        #             continue
        #         if len(j) == 4:
        #             tl += list(j[0].leftids) + list(j[1].leftids) + list(j[2].leftids) + list(j[3].leftids)
        #             tr += ['sre0' + i + '/' + k[:-6] + k[-2:] for k in
        #                    list(j[0].rightids) + list(j[1].rightids) + list(j[2].rightids) + list(j[3].rightids)]
        #         else:
        #             ul += list(j[0].leftids) + list(j[1].leftids)
        #             ur += [i + '/' + k[:-6] + k[-2:] for k in list(j[0].rightids) + list(j[1].rightids)]
        #     self.ubm_TV_idmap.leftids = np.array(tl + ul)
        #     self.ubm_TV_idmap.rightids = np.array(tr + ur)
        #     self.ubm_TV_idmap.start = np.empty(np.size(self.ubm_TV_idmap.leftids), '|O')
        #     self.ubm_TV_idmap.stop = np.empty(np.size(self.ubm_TV_idmap.leftids), '|O')
        #     self.plda_idmap.leftids = np.array(tl)
        #     self.plda_idmap.rightids = np.array(tr)
        #     self.plda_idmap.start = np.empty(np.size(self.plda_idmap.leftids), '|O')
        #     self.plda_idmap.stop = np.empty(np.size(self.plda_idmap.leftids), '|O')
        #     self.trial_key = trial_key
        #     self.trial_key.write(root + "fea/trial_key_custom.h5")
        #     self.test_ndx.write(root + "fea/test_ndx_custom.h5")
        #     self.enroll_idmap.write(root + "fea/enroll_idmap_custom.h5")
        #     self.test_idmap.write(root + "fea/test_idmap_custom.h5")
        #     self.ubm_TV_idmap.write(root + "fea/ubm_TV_idmap_custom.h5")
        #     self.plda_idmap.write(root + "fea/plda_idmap_custom.h5")
        # else:
        #     self.trial_key = sidekit.Key(root + "fea/trial_key_custom.h5")
        #     self.test_ndx = sidekit.Ndx(root + "fea/test_ndx_custom.h5")
        #     self.enroll_idmap = sidekit.IdMap(root + "fea/enroll_idmap_custom.h5")
        #     self.test_idmap = sidekit.IdMap(root + "fea/test_idmap_custom.h5")
        #     self.ubm_TV_idmap = sidekit.IdMap(root + "fea/ubm_TV_idmap_custom.h5")
        #     self.plda_idmap = sidekit.IdMap(root + "fea/plda_idmap_custom.h5")
        ###############################################################################
        self.trial_key = sidekit.Key(root + "fea/trial_key_" + saveFlag + ".h5")
        self.test_ndx = sidekit.Ndx(root + "fea/test_ndx_" + saveFlag + ".h5")
        self.enroll_idmap = sidekit.IdMap(root + "fea/enroll_idmap_" + saveFlag + ".h5")
        self.test_idmap = sidekit.IdMap(root + "fea/test_idmap_" + saveFlag + ".h5")

        self.ubm_TV_idmap = sidekit.IdMap(root + "fea/ubm_TV_idmap.h5")
        self.plda_idmap = sidekit.IdMap(root + "fea/plda_idmap.h5")
        # </editor-fold>

        # feature Server ###############################################################
        self.feaServer = FeaServer(feature_filename_structure=self.feature_dir + "{}.h5",
                                   dataset_list=["cep"])

    '''Step1: Create the universal background model from all the training speaker data'''

    @ut.timing("trainUBM")
    def trainUBM(self, data=None):
        self.ubm = Mixture()
        ubm_list = list(self.ubm_TV_idmap.rightids)
        llk = self.ubm.EM_split(self.feaServer, ubm_list, self.distrib_nb, num_thread=self.nbThread, save_partial=True,
                                output_file_name=ivectorWorkDir + 'ubm/ubm', llk_gain=0.005)
        self.ubm.write(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
        return self

    '''Step2.1: Calculate the statistics from train data set needed for the iVector model.'''

    @ut.timing("calStat")
    def calStat(self, data=None):
        if self.ubm == None:
            self.ubm = sidekit.Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
        # with ut.Timing("enroll_test"):
        #     feaS = copy.deepcopy(self.feaServer)
        #     features = feaS.stack_features_parallel(list(self.enroll_idmap.rightids),
        #                                             num_thread=self.nbThread)
        #     feaS.fea = dict(features)
        #
        #     enroll_stat = StatServer(self.enroll_idmap, self.distrib_nb, self.feaSize)
        #     enroll_stat.accumulate_stat(ubm=self.ubm, feature_server=feaS,
        #                                 seg_indices=range(enroll_stat.segset.shape[0]), num_thread=self.nbThread)
        #     enroll_stat.write(ivectorWorkDir + 'stat/enroll_"+saveFlag+"_{}.h5'.format(self.distrib_nb))
        #
        #     feaS = copy.deepcopy(self.feaServer)
        #     features = feaS.stack_features_parallel(list(self.test_idmap.rightids),
        #                                             num_thread=self.nbThread)
        #     feaS.fea = dict(features)
        #
        #     test_stat = StatServer(self.test_idmap, self.distrib_nb, self.feaSize)
        #     test_stat.accumulate_stat(ubm=self.ubm, feature_server=feaS,
        #                               seg_indices=range(test_stat.segset.shape[0]), num_thread=self.nbThread)
        #     test_stat.write(ivectorWorkDir + 'stat/test_"+saveFlag+"_{}.h5'.format(self.distrib_nb))
        with ut.Timing("calStat_enroll_test"):
            with ut.Timing("calStat_dataRead_enroll"):
                feaS = FeaServer(feature_filename_structure=self.feature_dir + "{}.h5",
                                 dataset_list=["cep"])
                feaS.fea = feaS.stack_features_parallel2(list(self.enroll_idmap.rightids),
                                                         num_thread=self.nbThread)

            self.enroll_stat = StatServer(self.enroll_idmap, self.distrib_nb, self.feaSize)
            self.enroll_stat.accumulate_stat(ubm=self.ubm, feature_server=feaS,
                                             seg_indices=range(self.enroll_stat.segset.shape[0]),
                                             num_thread=self.nbThread)
            self.enroll_stat.write(ivectorWorkDir + "stat/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            with ut.Timing("calStat_dataRead_test"):
                feaS = FeaServer(feature_filename_structure=self.feature_dir + "{}.h5",
                                 dataset_list=["cep"])
                feaS.fea = feaS.stack_features_parallel2(list(self.test_idmap.rightids),
                                                         num_thread=self.nbThread)

            self.test_stat = StatServer(self.test_idmap, self.distrib_nb, self.feaSize)
            self.test_stat.accumulate_stat(ubm=self.ubm, feature_server=feaS,
                                           seg_indices=range(self.test_stat.segset.shape[0]), num_thread=self.nbThread)
            self.test_stat.write(ivectorWorkDir + "stat/test_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        with ut.Timing("calStat_background"):
            with ut.Timing("calStat_dataRead_train"):
                if self.feaServer.fea == None:
                    feaS = FeaServer(feature_filename_structure=self.feature_dir + "{}.h5",
                                     dataset_list=["cep"])
                    feaS.fea = feaS.stack_features_parallel2(list(self.ubm_TV_idmap.rightids),
                                                             num_thread=self.nbThread)

            self.back_stat = StatServer(self.ubm_TV_idmap, self.distrib_nb, self.feaSize)
            self.back_stat.accumulate_stat(ubm=self.ubm, feature_server=feaS,
                                           seg_indices=range(self.back_stat.segset.shape[0]), num_thread=self.nbThread)
            self.back_stat.write(ivectorWorkDir + 'stat/train_{}.h5'.format(self.distrib_nb))
        # free memory
        self.feaServer.fea = None

        return self
    '''Step2.2: Learn the total variability subspace from all the train speaker data.'''

    @ut.timing("learnTV")
    def learnTV(self, data=None):
        if self.ubm == None:
            self.ubm = sidekit.Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
        with ut.Timing("learnTV_StatServer"):
            if self.back_stat == None:
                self.back_stat = StatServer(ivectorWorkDir + 'stat/train_{}.h5'.format(
                    self.distrib_nb))  # this is super class,but not child class
        # cause Tmatrix compute so complex,i only use 10000 utterances to train it
        self.back_stat.partition(10000)
        tv_mean, tv, _, __, tv_sigma = self.back_stat.factor_analysis(rank_f=self.rank_TV,
                                                                      rank_g=0,
                                                                      rank_h=None,
                                                                      re_estimate_residual=False,
                                                                      it_nb=(self.tv_iteration, 0, 0),
                                                                      min_div=True,
                                                                      ubm=self.ubm,
                                                                      batch_size=self.batchSize,
                                                                      num_thread=self.nbThread,
                                                                      save_partial=ivectorWorkDir + "Tmatrix/T_{}".format(
                                                                          self.distrib_nb))
        self.Tmatrix = [tv, tv_mean, tv_sigma]
        sidekit.sidekit_io.write_tv_hdf5((tv, tv_mean, tv_sigma),
                                         ivectorWorkDir + "Tmatrix/T_{}".format(self.distrib_nb))
        return self
    '''Step2.3:Now compute the development ivectors of train data set for each speaker and channel.  The result is size tvDim x nSpeakers x nChannels.'''

    @ut.timing("extractIV")
    def extractIV(self, data=None):
        # just extract which i want to use in future,e.g:enroll,test ,train portion in sre04-08,
        # so that i would not extract ivector on switchnboard,cause i just use sre04-08 to train lda and plda
        if self.Tmatrix == None:
            tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5(
                ivectorWorkDir + "Tmatrix/T_{}".format(self.distrib_nb))
        else:
            tv, tv_mean, tv_sigma = self.Tmatrix

        with ut.Timing("enroll_test"):
            if self.enroll_stat == None:
                self.enroll_stat = sidekit.StatServer(
                    ivectorWorkDir + "stat/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb))

            enroll_iv = \
                self.enroll_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=self.batchSize,
                                                 num_thread=self.nbThread)[0]
            enroll_iv.write(ivectorWorkDir + "iv/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            if self.test_stat == None:
                self.test_stat = sidekit.StatServer(
                    ivectorWorkDir + "stat/test_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            test_iv = \
                self.test_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=self.batchSize,
                                               num_thread=self.nbThread)[
                    0]
            test_iv.write(ivectorWorkDir + "iv/test_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        # cause that plda train data is not the same as before,so there must be read
        with ut.Timing("background"):
            self.back_stat = sidekit.StatServer.read_subset(ivectorWorkDir + "stat/train_{}.h5".format(self.distrib_nb),
                                                            self.plda_idmap)
            train_iv = \
                self.back_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=self.batchSize,
                                               num_thread=self.nbThread)[0]
            train_iv.write(ivectorWorkDir + "iv/train_{}.h5".format(self.distrib_nb))


        ##############################################################################################################
        ##############################################################################################################
        ##############################################################################################################
        ################################begin to calculate score######################################################
        ##############################################################################################################
        ##############################################################################################################
        ##############################################################################################################


    '''Step3.1:do LDA on the development iVectors to find the dimensions that matter.'''

    @ut.timing("LDA_WCCN_cos_Score")
    def LDA_WCCN_cos_Score(self, data=None):
        if self.train_iv == None:
            self.train_iv, self.enroll_iv, self.test_iv = self.readiv(["iv/train", "iv/enroll", "iv/test"])

        train_iv_wccn, enroll_iv_wccn, test_iv_wccn = self.deepcopy(self.train_iv, self.enroll_iv, self.test_iv)

        scores_cos = sidekit.iv_scoring.cosine_scoring(enroll_iv_wccn, test_iv_wccn, self.test_ndx, wccn=None)
        scores_cos.write(ivectorWorkDir + "score/cos_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###########################################################################
        wccn = train_iv_wccn.get_wccn_choleski_stat1()
        scores_cos_wccn = sidekit.iv_scoring.cosine_scoring(enroll_iv_wccn, test_iv_wccn, self.test_ndx, wccn=wccn)
        scores_cos_wccn.write(ivectorWorkDir + "score/cos_wccn_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###########################################################################
        if self.train_iv_lda == None:
            self.train_iv_lda, self.enroll_iv_lda, self.test_iv_lda = self.deepcopy(self.train_iv, self.enroll_iv,
                                                                                    self.test_iv)
            LDA = self.train_iv_lda.get_lda_matrix_stat1(self.lda_rk)
            self.train_iv_lda.rotate_stat1(LDA)
            self.enroll_iv_lda.rotate_stat1(LDA)
            self.test_iv_lda.rotate_stat1(LDA)

            self.train_iv_lda.write(ivectorWorkDir + "lda/train_{}.h5".format(self.distrib_nb))
            self.enroll_iv_lda.write(ivectorWorkDir + "lda/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            self.test_iv_lda.write(ivectorWorkDir + "lda/test_" + saveFlag + "_{}.h5".format(self.distrib_nb))

        scores_cos_lda = sidekit.iv_scoring.cosine_scoring(self.enroll_iv_lda, self.test_iv_lda, self.test_ndx,
                                                           wccn=None)
        scores_cos_lda.write(ivectorWorkDir + "score/cos_lda_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###########################################################################
        wccn = self.train_iv_lda.get_wccn_choleski_stat1()
        scores_cos_lda_wcnn = sidekit.iv_scoring.cosine_scoring(self.enroll_iv_lda, self.test_iv_lda, self.test_ndx,
                                                                wccn=wccn)
        scores_cos_lda_wcnn.write(ivectorWorkDir + "score/cos_lda_wccn_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        return self

    @ut.timing("mahalanobis_distance_Score")
    def mahalanobis_distance_Score(self):
        if self.train_iv == None:
            self.train_iv, self.enroll_iv, self.test_iv = self.readiv(["iv/train", "iv/enroll", "iv/test"])
        meanEFR, CovEFR = self.train_iv.estimate_spectral_norm_stat1(3)

        self.train_iv.spectral_norm_stat1(meanEFR[:1], CovEFR[:1])
        self.enroll_iv.spectral_norm_stat1(meanEFR[:1], CovEFR[:1])
        self.test_iv.spectral_norm_stat1(meanEFR[:1], CovEFR[:1])
        M1 = self.train_iv.get_mahalanobis_matrix_stat1()
        scores_mah_efr1 = sidekit.iv_scoring.mahalanobis_scoring(self.enroll_iv, self.test_iv, self.test_ndx, M1)
        scores_mah_efr1.write(ivectorWorkDir + "score/madis_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        return self

    @ut.timing("two_covariance_Score")
    def two_covariance_Score(self):
        if self.train_iv == None:
            self.train_iv, self.enroll_iv, self.test_iv = self.readiv(["iv/train", "iv/enroll", "iv/test"])
        train_iv1, enroll_iv1, test_iv1 = self.deepcopy(self.train_iv, self.enroll_iv, self.test_iv)

        W = train_iv1.get_within_covariance_stat1()
        B = train_iv1.get_between_covariance_stat1()
        scores_2cov = sidekit.iv_scoring.two_covariance_scoring(enroll_iv1, test_iv1, self.test_ndx, W, B)
        scores_2cov.write(ivectorWorkDir + "score/2covar_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ########################################################
        meanSN, CovSN = self.train_iv.estimate_spectral_norm_stat1(1, "sphNorm")

        self.train_iv.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        self.enroll_iv.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        self.test_iv.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        W1 = self.train_iv.get_within_covariance_stat1()
        B1 = self.train_iv.get_between_covariance_stat1()
        scores_2cov_sn1 = sidekit.iv_scoring.two_covariance_scoring(self.enroll_iv, self.test_iv, self.test_ndx, W1, B1)
        scores_2cov_sn1.write(ivectorWorkDir + "score/2covar_sphernorm_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        return self

    '''Step3.2: Now train a Gaussian PLDA model with development i-vectors'''

    @ut.timing("PLDA_Score")
    def PLDA_Score(self, data=None):
        # use  Spherical Nuisance Normalization,the length-norm is a case in this norm
        if self.train_iv == None:
            self.train_iv, self.enroll_iv, self.test_iv = self.readiv(["iv/train", "iv/enroll", "iv/test"])
        meanSN, CovSN = self.train_iv.estimate_spectral_norm_stat1(1,
                                                                   "sphNorm")  # default is length norm when 2nd parameter is "efr"
        self.train_iv.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        self.enroll_iv.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        self.test_iv.spectral_norm_stat1(meanSN[:1], CovSN[:1])

        plda_mean, plda_F, plda_G, plda_H, plda_Sigma = self.train_iv.factor_analysis(rank_f=self.plda_rk, rank_g=0,
                                                                                      rank_h=None,
                                                                                      re_estimate_residual=True,
                                                                                      it_nb=(10, 0, 0), min_div=True,
                                                                                      ubm=None,
                                                                                      batch_size=self.batchSize * 10,
                                                                                      num_thread=self.nbThread)
        sidekit.sidekit_io.write_plda_hdf5((plda_mean, plda_F, plda_G, plda_Sigma),
                                           ivectorWorkDir + "plda/plda_sphernorm_" + saveFlag + "_{}.h5".format(
                                               self.distrib_nb))
        scores_plda = sidekit.iv_scoring.PLDA_scoring(self.enroll_iv, self.test_iv, self.test_ndx, plda_mean, plda_F,
                                                      plda_G,
                                                      plda_Sigma, full_model=False)
        scores_plda.write(ivectorWorkDir + "score/plda_sphernorm_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        return self

    @ut.timing("PLDA_LDA_Score")
    def PLDA_LDA_Score(self, data=None):
        # use  Spherical Nuisance Normalization
        if self.train_iv_lda == None:
            self.train_iv_lda, self.enroll_iv_lda, self.test_iv_lda = self.readiv(
                ["lda/train", "lda/enroll", "lda/test"])

        train_iv1, enroll_iv1, test_iv1 = self.deepcopy(self.train_iv_lda, self.enroll_iv_lda, self.test_iv_lda)

        plda_mean, plda_F, plda_G, plda_H, plda_Sigma = train_iv1.factor_analysis(rank_f=self.lda_rk, rank_g=0,
                                                                                  rank_h=None,
                                                                                  re_estimate_residual=True,
                                                                                  it_nb=(10, 0, 0), min_div=True,
                                                                                  ubm=None,
                                                                                  batch_size=self.batchSize * 10,
                                                                                  num_thread=self.nbThread)
        sidekit.sidekit_io.write_plda_hdf5((plda_mean, plda_F, plda_G, plda_Sigma),
                                           ivectorWorkDir + "plda/plda_lda_" + saveFlag + "_{}.h5".format(
                                               self.distrib_nb))
        scores_plda = sidekit.iv_scoring.PLDA_scoring(enroll_iv1, test_iv1, self.test_ndx, plda_mean, plda_F, plda_G,
                                                      plda_Sigma, full_model=False)
        scores_plda.write(ivectorWorkDir + "score/plda_lda_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###############################################################################
        train_wccn, enroll_wccn, test_wccn = self.deepcopy(self.train_iv_lda, self.enroll_iv_lda, self.test_iv_lda)

        wccn = train_wccn.get_wccn_choleski_stat1()
        train_wccn.rotate_stat1(wccn)
        enroll_wccn.rotate_stat1(wccn)
        test_wccn.rotate_stat1(wccn)

        train_wccn1, enroll_wccn1, test_wccn1 = self.deepcopy(train_wccn, enroll_wccn, test_wccn)

        train_wccn.write(ivectorWorkDir + "lda/train_wccn_{}.h5".format(self.distrib_nb))
        enroll_wccn.write(ivectorWorkDir + "lda/enroll_wccn_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        test_wccn.write(ivectorWorkDir + "lda/test_wccn_" + saveFlag + "_{}.h5".format(self.distrib_nb))

        plda_mean, plda_F, plda_G, plda_H, plda_Sigma = train_wccn.factor_analysis(rank_f=self.lda_rk, rank_g=0,
                                                                                   rank_h=None,
                                                                                   re_estimate_residual=True,
                                                                                   it_nb=(10, 0, 0), min_div=True,
                                                                                   ubm=None,
                                                                                   batch_size=self.batchSize * 10,
                                                                                   num_thread=self.nbThread)
        sidekit.sidekit_io.write_plda_hdf5((plda_mean, plda_F, plda_G, plda_Sigma),
                                           ivectorWorkDir + "plda/plda_lda_wccn_" + saveFlag + "_{}.h5".format(
                                               self.distrib_nb))
        scores_plda = sidekit.iv_scoring.PLDA_scoring(enroll_wccn, test_wccn, self.test_ndx, plda_mean, plda_F, plda_G,
                                                      plda_Sigma, full_model=False)
        scores_plda.write(ivectorWorkDir + "score/plda_lda_wccn_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###############################################################################
        meanSN, CovSN = train_wccn1.estimate_spectral_norm_stat1(1, "sphNorm")
        train_wccn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        enroll_wccn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        test_wccn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])

        plda_mean, plda_F, plda_G, plda_H, plda_Sigma = train_wccn1.factor_analysis(rank_f=self.plda_rk, rank_g=0,
                                                                                    rank_h=None,
                                                                                    re_estimate_residual=True,
                                                                                    it_nb=(10, 0, 0), min_div=True,
                                                                                    ubm=None,
                                                                                    batch_size=self.batchSize * 10,
                                                                                    num_thread=self.nbThread)
        sidekit.sidekit_io.write_plda_hdf5((plda_mean, plda_F, plda_G, plda_Sigma),
                                           ivectorWorkDir + "plda/plda_lda_wccn__sphernorm_" + saveFlag + "_{}.h5".format(
                                               self.distrib_nb))
        scores_plda = sidekit.iv_scoring.PLDA_scoring(enroll_wccn1, test_wccn1, self.test_ndx, plda_mean, plda_F,
                                                      plda_G,
                                                      plda_Sigma, full_model=False)
        scores_plda.write(
            ivectorWorkDir + "score/plda_lda_wccn_sphernorm_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###############################################################################
        meanSN, CovSN = self.train_iv_lda.estimate_spectral_norm_stat1(1, "sphNorm")
        self.train_iv_lda.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        self.enroll_iv_lda.spectral_norm_stat1(meanSN[:1], CovSN[:1])
        self.test_iv_lda.spectral_norm_stat1(meanSN[:1], CovSN[:1])

        plda_mean, plda_F, plda_G, plda_H, plda_Sigma = self.train_iv_lda.factor_analysis(rank_f=self.plda_rk, rank_g=0,
                                                                                          rank_h=None,
                                                                                          re_estimate_residual=True,
                                                                                          it_nb=(10, 0, 0),
                                                                                          min_div=True,
                                                                                          ubm=None,
                                                                                          batch_size=self.batchSize * 10,
                                                                                          num_thread=self.nbThread)
        sidekit.sidekit_io.write_plda_hdf5((plda_mean, plda_F, plda_G, plda_Sigma),
                                           ivectorWorkDir + "plda/plda_lda_sphernorm_" + saveFlag + "_{}.h5".format(
                                               self.distrib_nb))
        scores_plda = sidekit.iv_scoring.PLDA_scoring(self.enroll_iv_lda, self.test_iv_lda, self.test_ndx, plda_mean,
                                                      plda_F, plda_G,
                                                      plda_Sigma, full_model=False)
        scores_plda.write(ivectorWorkDir + "score/plda_lda_sphernorm_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        return self

    # there has no step4,because i merge it to past steps
    def computeIVAndDoLDA_OnEnrollOrTest(self, data=None):
        pass

    '''Step5: Now score the models with all the test data.'''

    @ut.timing("visualization")
    def visualization(self, data=None):
        # Set the prior parameters following NIST-SRE 2010 settings
        prior = sidekit.logit_effective_prior(0.001, 1, 1)
        # Initialize the DET plot to 2010 settings
        dp = sidekit.DetPlot(window_style="sre10", plot_title="IVector SRE 2010")
        dp.create_figure()

        graphData = ["cos", "cos_wccn", "cos_lda", "cos_lda_wccn", "madis", "2covar", "2covar_sphernorm",
                     "plda_sphernorm", "plda_lda", "plda_lda_wccn", "plda_lda_wccn_sphernorm", "plda_lda_sphernorm"]
        for j, i in enumerate(graphData):
            tmp = ivectorWorkDir + "score/" + i + "_" + saveFlag + "_{}.h5".format(self.distrib_nb)
            if os.path.exists(tmp):
                score = sidekit.Scores(tmp)
                dp.set_system_from_scores(score, self.trial_key, sys_name=i)
                dp.plot_rocch_det(j)

        dp.plot_DR30_both(idx=0)
        dp.plot_mindcf_point(prior, idx=0)

    def readiv(self, fileList: list = None) -> (sidekit.StatServer, sidekit.StatServer, sidekit.StatServer):
        train_iv = sidekit.StatServer.read(ivectorWorkDir + fileList[0] + "_{}.h5".format(self.distrib_nb))
        enroll_iv = sidekit.StatServer.read(
            ivectorWorkDir + fileList[1] + "_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        test_iv = sidekit.StatServer.read(
            ivectorWorkDir + fileList[2] + "_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        return train_iv, enroll_iv, test_iv

    def deepcopy(self, train_iv, enroll_iv, test_iv):
        train_iv_wccn = copy.deepcopy(train_iv)
        enroll_iv_wccn = copy.deepcopy(enroll_iv)
        test_iv_wccn = copy.deepcopy(test_iv)
        return train_iv_wccn, enroll_iv_wccn, test_iv_wccn

    def test(self):
        s = self.feaServer.load("sre10/10sec/1/lfwsgTB1", input_feature_filename=None, label=None, start=None,
                                stop=None)
        pass

def main():
    # fea=FeaServer(dataset_list=["ssss"])
    # fea.stack_features_parallel(["ss","qq","cc"],num_thread=3)
    # if True:
    #     return 0

    logger.info("=================================================================================================")
    iv = IV()
    iv.trainUBM().calStat().learnTV().extractIV()
    pass

    logger.info("=================================================================================================")
    # iv = IV()
    iv.LDA_WCCN_cos_Score().mahalanobis_distance_Score().two_covariance_Score().PLDA_Score().PLDA_LDA_Score().visualization()
    pass

if __name__ == '__main__':
    # shows=['tcueTA.h5','tcucTA.h5','laapkTA1.h5','laayvTA1.h5']
    # for show in shows[2:]:
    #     feat=[]
    #     h5f = h5py.File("/home/jyh/share/"+show, "r")
    #     show=re.findall("([a-z]{4,})",show)[0]
    #     feat.append(h5f["/".join(("10sec/1/"+show, "cep"))])
    #
    #     h5f.close()


    main()
