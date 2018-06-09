# #!/usr/bin/env python
# # -*- coding:utf-8 -*-
# import multiprocessing
# import h5py
#
# class f():
#     @staticmethod
#     def func(s,ss):
#         data[str(s)] = s
#
#     @staticmethod
#     def init(lock_, data_):
#         global lock
#         lock = lock_
#         global data
#         data = data_
#
#     @staticmethod
#     def main():
#         h5f = h5py.File("./f.txt", "w")
#         ma = multiprocessing.Manager()
#         data = ma.dict()
#         lock = multiprocessing.Lock()
#         po = multiprocessing.Pool(2, initargs=(lock, data), initializer=f.init)
#         for i in range(-20,-1):
#             po.apply_async(f.func, (i,i))
#         po.close()
#         po.join()
#         for i, j in data.items():
#             h5f[i] = j
#         h5f.close()
#
# def func(s):
#     data[str(s)]=s
#
# def init(lock_,data_):
#     global lock
#     lock=lock_
#     global data
#     data=data_
#
# def main():
#     h5f = h5py.File("./f.txt", "w")
#     ma=multiprocessing.Manager()
#     data=ma.dict()
#     lock=multiprocessing.Lock()
#     po=multiprocessing.Pool(2,initargs=(lock,data),initializer=init)
#     for i in range(20):
#         po.apply_async(func,(i,))
#     po.close()
#     po.join()
#     for i,j in data.items():
#         h5f[i]=j
#     h5f.close()
# if __name__ == '__main__':
#     # h5f = h5py.File("./f.txt", "w")
#     # for s in range(20):
#     #     h5f[str(s)] = s
#     # h5f.close()
#     f.main()
#     h5f = h5py.File("./f.txt", "r")
#     for i in [key for key in h5f.keys()]:
#         print(h5f[i].value)

#!/usr/bin/env python
# -*- coding:utf-8 -*-
#################################################################################
# In terms of paper 25:Deep Speaker Embeddings for Short-Duration Speaker Verification,
# NIST evaluations (2004-08) and a portion of the Switchboard dataset for training both
# deep networks and baseline i-vector/PLDA systems.
#
# Only use NIST-SRE 2010 female test set to test performance,but i will use all data available from 2010
# to be the test set

# other option:Paper33:X-VECTORS: ROBUST DNN EMBEDDINGS FOR SPEAKER RECOGNITION
# In the experiments, the extractors (UBM/T or embedding DNN) are trained on SWBD and SRE,
# and the PLDA classifiers are trained on just SRE.
################################################################################
# 'extract_ivectors' ,'total_variability' is in sidekit.factor_analyser.FactorAnalyser.
# 'EM_split' is in Mixture.
# all of these three method are also in sidekit.sidekit_mpi
################################################################################

import logging
import os
import re
import sys
import h5py
import csv

import time

os.environ["SIDEKIT"] = "theano=false,theano_config=cpu,libsvm=false,mpi=false"

# from sidekit.sidekit_mpi import EM_split, extract_ivector as mpi_extractIV, total_variability as mpi_learnTV
from mpiIV import EM_split as mpi_EM_split, extract_ivector as mpi_extractIV, total_variability as mpi_learnTV
from sidekit.frontend import cep_sliding_norm
from sidekit.mixture import sum_log_probabilities
from sidekit.sidekit_wrappers import process_parallel_lists
from sidekit.statserver import compute_llk, ct
import sidekit
from sidekit import sidekit_io, FeaturesServer, FactorAnalyser
import copy
import numpy as np, numpy
import IVector_Base
import jyh.Utils as ut
import h5py
import globalVar as glb
import multiprocessing
import warnings
import ctypes
from mpi4py import MPI
from sidekit.sv_utils import serialize
import scipy
from sidekit.factor_analyser import e_gather, e_worker

root = glb.get_root()  # '/home/jyh/D/jyh/data/'
experimentsIdentify = 'ivector_all'
inputDir = root + 'fea/'  # fea and idmap input dir
ivectorWorkDir = inputDir + experimentsIdentify + '/'  # output of ivector ,lda,plda score result and so on
logger = ut.logByLogginModule(experimentsIdentify)
saveFlag = "10s"
STAT_TYPE = np.float64
trainNum = -1  # -1 represent work with all data


class FA(FactorAnalyser):
    def __init__(self):
        super(FA, self).__init__()

    def total_variability(self,
                          stat_server_filename,
                          ubm,
                          tv_rank,
                          nb_iter=20,
                          min_div=True,
                          tv_init=None,
                          batch_size=300,
                          save_init=False,
                          output_file_name=None,
                          num_thread=1):

        if not isinstance(stat_server_filename, list):
            stat_server_filename = [stat_server_filename]

        assert (isinstance(ubm, Mixture) and ubm.validate()), "Second argument must be a proper Mixture"
        assert (isinstance(nb_iter, int) and (0 < nb_iter)), "nb_iter must be a positive integer"

        gmm_covariance = "diag" if ubm.invcov.ndim == 2 else "full"

        # Set useful variables
        with h5py.File(stat_server_filename[0], 'r') as fh:  # open the first StatServer to get size
            _, sv_size = fh['stat1'].shape
            feature_size = fh['stat1'].shape[1] // fh['stat0'].shape[1]
            distrib_nb = fh['stat0'].shape[1]

        upper_triangle_indices = numpy.triu_indices(tv_rank)

        # mean and Sigma are initialized at ZEROS as statistics are centered
        self.mean = numpy.zeros(ubm.get_mean_super_vector().shape, dtype=STAT_TYPE)
        self.F = serialize(numpy.zeros((sv_size, tv_rank)).astype(STAT_TYPE))
        if tv_init is None:
            self.F = numpy.random.randn(sv_size, tv_rank).astype(STAT_TYPE)
        else:
            self.F = tv_init
        self.Sigma = numpy.zeros(ubm.get_mean_super_vector().shape, dtype=STAT_TYPE)

        # Save init if required
        if output_file_name is None:
            output_file_name = "temporary_factor_analyser"
        if save_init:
            self.write(output_file_name + "_init.h5")

        # Estimate  TV iteratively
        for it in range(nb_iter):
            with ut.Timing("TV_EM:" + str(it)):
                # Create serialized accumulators for the list of models to process
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    _A = serialize(numpy.zeros((distrib_nb, tv_rank * (tv_rank + 1) // 2), dtype=STAT_TYPE))
                    _C = serialize(numpy.zeros((tv_rank, sv_size), dtype=STAT_TYPE))
                    _R = serialize(numpy.zeros((tv_rank * (tv_rank + 1) // 2), dtype=STAT_TYPE))

                total_session_nb = 0

                # E-step
                # Accumulate statistics for each StatServer from the list
                for stat_server_file in stat_server_filename:

                    # get info from the current StatServer
                    with h5py.File(stat_server_file, 'r') as fh:
                        nb_sessions = fh["modelset"].shape[0]
                        total_session_nb += nb_sessions
                        batch_nb = int(numpy.floor(nb_sessions / float(batch_size) + 0.999))
                        batch_indices = numpy.array_split(numpy.arange(nb_sessions), batch_nb)

                        manager = multiprocessing.Manager()
                        q = manager.Queue()
                        pool = multiprocessing.Pool(num_thread + 2)

                        # put Consumer to work first
                        watcher = pool.apply_async(e_gather, ((_A, _C, _R), q))
                        # fire off workers
                        jobs = []

                        # Load data per batch to reduce the memory footprint
                        for batch_idx in batch_indices:
                            # Create list of argument for a process
                            arg = fh["stat0"][batch_idx, :], fh["stat1"][batch_idx, :], ubm, self.F
                            job = pool.apply_async(e_worker, (arg, q))
                            jobs.append(job)

                        # collect results from the workers through the pool result queue
                        for job in jobs:
                            job.get()

                        # now we are done, kill the consumer
                        q.put((None, None, None, None))
                        pool.close()

                        _A, _C, _R = watcher.get()

                _R /= total_session_nb

                # M-step
                _A_tmp = numpy.zeros((tv_rank, tv_rank), dtype=STAT_TYPE)
                for c in range(distrib_nb):
                    distrib_idx = range(c * feature_size, (c + 1) * feature_size)
                    _A_tmp[upper_triangle_indices] = _A_tmp.T[upper_triangle_indices] = _A[c, :]
                    self.F[distrib_idx, :] = scipy.linalg.solve(_A_tmp, _C[:, distrib_idx]).T

                # Minimum divergence
                if min_div:
                    _R_tmp = numpy.zeros((tv_rank, tv_rank), dtype=STAT_TYPE)
                    _R_tmp[upper_triangle_indices] = _R_tmp.T[upper_triangle_indices] = _R
                    ch = scipy.linalg.cholesky(_R_tmp)
                    self.F = self.F.dot(ch)

                # Save the current FactorAnalyser
                if output_file_name is not None:
                    if it < nb_iter - 1:
                        self.write(output_file_name + "_it-{}.h5".format(it))
                    else:
                        self.write(output_file_name + ".h5")


class FeaServer(sidekit.FeaturesServer):
    def __init__(self, feature_filename_structure,
                 dataset_list=["cep"],
                 mask=None,
                 feat_norm=None,
                 # cmvn has not do ,so there should do it,but o\i do not use post processe to handle it ,
                 # so i work it with overwrite method
                 #  FeaturesServer.get_features()
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

    #    self.fea = None  # dict(show:feature)

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
    #     return dict(output)  # np.concatenate(output, axis=0)

    def get_features(self, show, channel=0, input_feature_filename=None, label=None, start=None, stop=None):
        if input_feature_filename is not None:
            self.feature_filename_structure = input_feature_filename

        # If no extractor for this source, open hdf5 file and return handler
        if self.features_extractor is None:
            h5f = h5py.File(self.feature_filename_structure.format(show), "r")

        # If an extractor is provided for this source, extract features and return an hdf5 handler
        else:
            h5f = self.features_extractor.extract(show, channel, input_audio_filename=input_feature_filename)

        if show[:5] == 'sre10':
            show = re.findall('sre10/(.*)[T|M][A|B]', show)[0]
        else:
            show = re.findall('/([^/]{4,})[T|M][A|B]', show)[0]

        feat = []
        if "cep" in self.dataset_list:
            cep = h5f["/".join((show, "cep"))].value
            cep = FeaServer.cmvn(cep)
            # cep = cep_sliding_norm(cep, label=None, win=301, center=True, reduce=True)
            feat.append(cep)
        if "fb" in self.dataset_list:
            fbank = h5f["/".join((show, "fb"))].value
            fbank = FeaServer.cmvn(fbank)
            # fbank = cep_sliding_norm(fbank, label=None, win=301, center=True, reduce=True)
            feat.append(fbank)
        feat = np.hstack(feat)
        label = numpy.ones(feat.shape[0], dtype='bool')
        h5f.close()
        return feat, label


        # if "energy" in self.dataset_list:
        #     feat.append(h5f["/".join((show, "energy"))].value[start:stop, numpy.newaxis])
        #     global_mean.append(h5f["/".join((show, "energy_mean"))].value)
        #     global_std.append(h5f["/".join((show, "energy_std"))].value)
        # if "cep" in self.dataset_list:
        #     feat.append(h5f["/".join((show, "cep"))][start:stop, :])
        #     global_mean.append(h5f["/".join((show, "cep_mean"))].value)
        #     global_std.append(h5f["/".join((show, "cep_std"))].value)
        # if "fb" in self.dataset_list:
        #     feat.append(h5f["/".join((show, "fb"))][start:stop, :])
        #     global_mean.append(h5f["/".join((show, "fb_mean"))].value)
        #     global_std.append(h5f["/".join((show, "fb_std"))].value)
        # if "bnf" in self.dataset_list:
        #     feat.append(h5f["/".join((show, "bnf"))][start:stop, :])
        #     global_mean.append(h5f["/".join((show, "bnf_mean"))].value)
        #     global_std.append(h5f["/".join((show, "bnf_std"))].value)
        # feat = numpy.hstack(feat)
        # global_mean = numpy.hstack(global_mean)
        # global_std = numpy.hstack(global_std)
        #
        # if label is None:
        #     if "/".join((show, "vad")) in h5f:
        #         label = h5f.get("/".join((show, "vad"))).value.astype('bool').squeeze()[start:stop]
        #     else:
        #         label = numpy.ones(feat.shape[0], dtype='bool')
        # # Pad the segment if needed
        # feat = numpy.pad(feat, ((pad_begining, pad_end), (0, 0)), mode='edge')
        # label = numpy.pad(label, (pad_begining, pad_end), mode='edge')
        # stop += pad_begining + pad_end
        #
        # h5f.close()
        # # Post-process the features and return the features and vad label
        # if global_cmvn:
        #     feat, label = self.post_processing(feat, label, global_mean, global_std)
        # else:
        #     feat, label = self.post_processing(feat, label)

        # return feat, label






        # if input_feature_filename is not None:
        #     self.feature_filename_structure = input_feature_filename
        #
        # h5f = h5py.File(self.feature_filename_structure.format(show), "r")
        #
        # show_ = copy.deepcopy(show)
        # if show[:5] == 'sre10':
        #     show = re.findall('sre10/(.*)[T|M][A|B]', show)[0]
        # else:
        #     show = re.findall('/([^/]{4,})[T|M][A|B]', show)[0]
        #
        # feat = []
        #
        # if "cep" in self.dataset_list:
        #     cep = h5f["/".join((show, "cep"))].value
        #     cep = FeaServer.cmvn(cep)
        #     # cep = cep_sliding_norm(cep, label=None, win=301, center=True, reduce=True)
        #     feat.append(cep)
        # if "fb" in self.dataset_list:
        #     fbank = h5f["/".join((show, "fb"))].value
        #     fbank = FeaServer.cmvn(fbank)
        #     # fbank = cep_sliding_norm(fbank, label=None, win=301, center=True, reduce=True)
        #     feat.append(fbank)
        # feat = np.hstack(feat)
        # h5f.close()
        #
        # return [[show_, feat]]

    # def stack_features_parallel2(self, show_list, num_thread=1, returnDictOrFeature=True):
    #     featuresL, labelL = FeaServer.multiReadProc(show_list, num_thread, self.feature_filename_structure,
    #                                                 self.dataset_list)
    #
    #     return dict(zip(labelL, featuresL)) if returnDictOrFeature else np.concatenate(featuresL, axis=0)
    #
    # @staticmethod
    # def multiReadProc(shows, num_thread, feature_filename_structure, dataset_list):
    #     lens = len(shows)
    #     manager = multiprocessing.Manager()
    #     fea_ = manager.list([0] * lens)
    #     label_ = manager.list([0] * lens)
    #     lock_ = manager.Lock()
    #     cou = manager.list([0])
    #     feature_filename_structure, dataset_list = list([feature_filename_structure]) * lens, list(
    #         [dataset_list]) * lens
    #     pool = multiprocessing.Pool(num_thread, initializer=FeaServer.globalVarinit,
    #                                 initargs=(lock_, fea_, label_, cou))
    #     pool.map(FeaServer.proc, zip(shows, list(range(lens)), feature_filename_structure, dataset_list))
    #     pool.close()
    #     pool.join()
    #     return fea_, label_
    #
    # @staticmethod
    # def globalVarinit(_lock, _fea, _label, co):
    #     global fea_
    #     global label_
    #     global lock_
    #     global cou
    #     cou = co
    #     # global feature_filename_structure_
    #     # global dataset_list_
    #     # feature_filename_structure_ = _feature_filename_structure
    #     # dataset_list_ = _dataset_list
    #     label_ = _label
    #     fea_ = _fea
    #     lock_ = _lock
    #
    # @staticmethod
    # def proc(show_):
    #     sho = show_[0]
    #     ind = show_[1]
    #     feature_filename_structure_ = show_[2]
    #     dataset_list_ = show_[3]
    #     h5f = h5py.File(feature_filename_structure_.format(sho), "r")
    #
    #     show1_ = sho
    #     if sho[:5] == 'sre10':
    #         sho = re.findall('sre10/(.*)[T|M][A|B]', sho)[0]
    #     else:
    #         sho = re.findall('/([^/]{4,})[T|M][A|B]', sho)[0]
    #
    #     feat = []
    #     # postProcessing:normalization
    #
    #     if "cep" in dataset_list_:
    #         cep = h5f["/".join((sho, "cep"))].value
    #         cep = FeaServer.cmvn(cep)
    #         # cep = cep_sliding_norm(cep, label=None, win=301, center=True, reduce=True)
    #         feat.append(cep)
    #     if "fb" in dataset_list_:
    #         fbank = h5f["/".join((sho, "fb"))].value
    #         fbank = FeaServer.cmvn(fbank)
    #         # fbank = cep_sliding_norm(fbank, label=None, win=301, center=True, reduce=True)
    #         feat.append(fbank)
    #     feat = np.hstack(feat)
    #     h5f.close()
    #     #################normalization:
    #
    #     with lock_:
    #         fea_[ind] = feat
    #         label_[ind] = show1_
    #         cou[0] += 1
    #         print(cou[0])

    @staticmethod
    def cmvn(features):
        mu = np.mean(features, axis=0)
        stdev = np.std(features, axis=0)
        features -= mu
        features /= stdev
        return features


class Mixture(sidekit.Mixture):
    def __init__(self, mixture_file_name=''):
        super(Mixture, self).__init__(mixture_file_name=mixture_file_name)

    def EM_split(self,
                 features_server,
                 feature_list,
                 distrib_nb,
                 iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
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
        for it in iterations[:int(numpy.log2(distrib_nb))]:
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
                llk_acc = numpy.zeros(1)
                sh = llk_acc.shape
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    tmp = multiprocessing.Array(ctypes.c_double, llk_acc.size)
                    llk_acc = numpy.ctypeslib.as_array(tmp.get_obj())
                    llk_acc = llk_acc.reshape(sh)

                with ut.Timing('ubm Expectation and Maximization:' + str(i)):
                    # E step
                    self._expectation_list(stat_acc=accum,
                                           feature_list=feature_list,
                                           feature_server=features_server,
                                           llk_acc=llk_acc,
                                           num_thread=num_thread)
                    llk.append(llk_acc[0] / numpy.sum(accum.w))

                    # M step
                    self._maximization(accum, ceil_cov=ceil_cov, floor_cov=floor_cov)
                if i > 0:
                    gain = llk[-1] - llk[-2]

                    if gain < llk_gain:
                        logger.info('EM (should break) distrib_nb: %d %i/%d gain: %f' % (
                            self.mu.shape[0], i + 1, it, gain))
                        continue
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

    def _init(self, features_server, feature_list, num_thread=1):
        # #############save feature and fileId#############################
        # features_server.fea = features_server.stack_features_parallel(feature_list[:5], num_thread=num_thread)
        # #################################################################
        # features_server.fea = features_server.stack_features_parallel2(feature_list, num_thread=num_thread)
        ####################################################################
        # Init using all data
        with ut.Timing("_init() getFeature"):
            features = features_server.stack_features_parallel(feature_list, num_thread=num_thread)
        with ut.Timing("_init() other operate"):
            n_frames = features.shape[0]
            mu = features.mean(0)
            cov = (features ** 2).mean(0)

            # n_frames, mu, cov = mean_std_many(features_server, feature_list, in_context=False, num_thread=num_thread)
            self.mu = mu[None]
            self.invcov = 1. / cov[None]
            self.w = numpy.asarray([1.0])
            self.cst = numpy.zeros(self.w.shape)
            self.det = numpy.zeros(self.w.shape)
            self.cov_var_ctl = 1.0 / copy.deepcopy(self.invcov)
            self._compute_all()


            #
            # with ut.Timing("_init() getFeature"):
            #     feaDict = Mixture.getFea(featuresDir,feature_list)
            #
            # with ut.Timing("_init() other operate"):
            #     fea = np.concatenate(list(feaDict.values()), axis=0)
            #     mu = fea.mean(0)
            #     cov = (fea ** 2).mean(0)
            #     # free room
            #     del fea
            #     # n_frames, mu, cov = mean_std_many(features_server, feature_list, in_context=False, num_thread=num_thread)
            #     self.mu = mu[None]
            #     self.invcov = 1. / cov[None]
            #     self.w = np.asarray([1.0])
            #     self.cst = np.zeros(self.w.shape)
            #     self.det = np.zeros(self.w.shape)
            #     self.cov_var_ctl = 1.0 / copy.deepcopy(self.invcov)
            #     self._compute_all()

    # @staticmethod
    # def getFea(featuresDir,feature_list=None):
    #     if feature_list[0][:5]=="sre10":
    #         feature_list = ["sre10_"+i[-8:] for i in feature_list]
    #     else:
    #         feature_list=[re.sub("/","_",i) for i in feature_list]
    #     fea = {}
    #     with h5py.File(featuresDir, 'r') as f:
    #         for j,i in enumerate(feature_list):
    #             fea[i]=f[i].value#.astype('float16')
    #             if j % 100 == 0 and j != 0:
    #                 print(j)
    #                 # break
    #     #     for j,i in enumerate(f.keys()):
    #     #         fea[i] = f[i].value
    #     #         if j%100==0 and j!=0:
    #     #             print(j)
    #     #         # if j==nums:
    #     #         #     break
    #
    #     return fea

    # use to mpi UBM calculate,just remove parallel compute.
    def _expectation_list(self, stat_acc, feature_list, feature_server, llk_acc=numpy.zeros(1), num_thread=1):
        # stat_acc._reset()#cause that has do it in mpiIV
        feature_server.keep_all_features = False
        for feat in feature_list:
            cep = feature_server.load(feat)[0]
            llk_acc[0] += self._expectation(stat_acc, cep)
        return llk_acc


class StatServer(sidekit.StatServer):
    '''
    both the method 'accumulate_stat' and 'accumulate_stat1' can calculate stat,i have try and debug both them are right.

    '''

    def __init__(self, statserver_file_name=None, distrib_nb=0, feature_size=0, index=None):
        super(StatServer, self).__init__(statserver_file_name, distrib_nb, feature_size, index)
        # self.modelset = np.empty(0, dtype="|O")
        # self.segset = np.empty(0, dtype="|O")
        # self.start = np.empty(0, dtype="|O")
        # self.stop = np.empty(0, dtype="|O")
        # self.stat0 = np.array([], dtype=STAT_TYPE)
        # self.stat1 = np.array([], dtype=STAT_TYPE)
        #
        # if isinstance(statserver_file_name, str) and index is None:
        #     tmp = StatServer.read(statserver_file_name)
        #     self.modelset = tmp.modelset
        #     self.segset = tmp.segset
        #     self.start = tmp.start
        #     self.stop = tmp.stop
        #     self.stat0 = tmp.stat0
        #     self.stat1 = tmp.stat1
        #
        # elif isinstance(statserver_file_name, sidekit.IdMap):
        #     self.modelset = statserver_file_name.leftids
        #     self.segset = statserver_file_name.rightids
        #     self.start = statserver_file_name.start
        #     self.stop = statserver_file_name.stop
        #     self.stat0 = multiprocessing.Manager().list()
        #     self.stat1 = multiprocessing.Manager().list()

    def partition(self, part):
        self.modelset = self.modelset[:part]
        self.segset = self.segset[:part]
        self.start = self.start[:part]
        self.stop = self.stop[:part]
        self.stat0 = self.stat0[:part]
        self.stat1 = self.stat1[:part]

    @staticmethod
    def proc(show_idx):
        sho = show_idx[0]
        cep = server_.load(sho)[0]
        # if sho[:5]=="sre10":
        #     sho = "sre10_"+sho[-8:]
        # else:
        #     sho=re.sub("/","_",sho)
        # cep = feaDict_[sho]
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
                cou_[0] += 1
                print(cou_[0])

    @staticmethod
    def multiReadProc(shows, num_thread, ubm, server):
        lens = len(shows)
        manager = multiprocessing.Manager()
        stat0_ = manager.list([0] * lens)
        stat1_ = manager.list([0] * lens)
        cou_ = manager.list([0])
        lock_ = manager.Lock()

        pool = multiprocessing.Pool(num_thread, initializer=StatServer.globalVarinit,
                                    initargs=(
                                        lock_, stat0_, stat1_, ubm,
                                        server, cou_))  # default number of processes is os.cpu_count()
        pool.map(StatServer.proc, zip(shows, list(range(lens))))
        pool.close()
        pool.join()
        return stat0_, stat1_

    @staticmethod
    def globalVarinit(_lock, _data, _data1, _ubm, _server, _cou):
        global stat0_
        global stat1_
        global lock_
        global ubm_
        global server_
        global cou_
        cou_ = _cou
        ubm_ = _ubm
        server_ = _server
        stat1_ = _data1
        stat0_ = _data
        lock_ = _lock

    # use my own code
    # @process_parallel_lists
    def accumulate_stat1(self, ubm, feature_server, seg_indices=None, channel_extension=("", "_b"), num_thread=1):
        stat0, stat1 = StatServer.multiReadProc(list(self.segset), num_thread, ubm, feature_server)
        self.stat0 = np.array(stat0)
        self.stat1 = np.array(stat1)

    @process_parallel_lists
    def accumulate_stat(self, ubm, feature_server, seg_indices=None, channel_extension=("", "_b"), num_thread=1):
        assert isinstance(ubm, Mixture), 'First parameter has to be a Mixture'
        assert isinstance(feature_server, FeaturesServer), 'Second parameter has to be a FeaturesServer'

        if (seg_indices is None) \
                or (self.stat0.shape[0] != self.segset.shape[0]) \
                or (self.stat1.shape[0] != self.segset.shape[0]):
            self.stat0 = numpy.zeros((self.segset.shape[0], ubm.distrib_nb()), dtype=STAT_TYPE)
            self.stat1 = numpy.zeros((self.segset.shape[0], ubm.sv_size()), dtype=STAT_TYPE)
            seg_indices = range(self.segset.shape[0])
        feature_server.keep_all_features = True

        for count, idx in enumerate(seg_indices):
            # logging.debug('Compute statistics for {}'.format(self.segset[idx]))
            # glb.set_count()
            show = self.segset[idx]

            # If using a FeaturesExtractor, get the channel number by checking the extension of the show
            channel = 0
            if feature_server.features_extractor is not None and show.endswith(channel_extension[1]):
                channel = 1
            show = show[:show.rfind(channel_extension[channel])]

            cep, vad = feature_server.load(show, channel=channel)
            stop = vad.shape[0] if self.stop[idx] is None else min(self.stop[idx], vad.shape[0])
            # logging.info('{} start: {} stop: {}'.format(show, self.start[idx], stop))
            data = cep[self.start[idx]:stop, :]
            data = data[vad[self.start[idx]:stop], :]

            # Verify that frame dimension is equal to gmm dimension
            if not ubm.dim() == data.shape[1]:
                raise Exception('dimension of ubm and features differ: {:d} / {:d}'.format(ubm.dim(), data.shape[1]))
            else:
                if ubm.invcov.ndim == 2:
                    lp = ubm.compute_log_posterior_probabilities(data)
                else:
                    lp = ubm.compute_log_posterior_probabilities_full(data)
                pp, foo = sum_log_probabilities(lp)
                # Compute 0th-order statistics
                self.stat0[idx, :] = pp.sum(0)
                # Compute 1st-order statistics
                self.stat1[idx, :] = numpy.reshape(numpy.transpose(
                    numpy.dot(data.transpose(), pp)), ubm.sv_size()).astype(STAT_TYPE)

                # glb.reset_count()  #reset countor

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
    def __init__(self, nbThread):
        self.createDir()
        self.ubm = None
        self.back_stat = None
        self.enroll_stat = None
        self.test_stat = None
        self.Tmatrix = None
        self.train_iv, self.enroll_iv, self.test_iv = None, None, None
        self.train_iv_lda, self.enroll_iv_lda, self.test_iv_lda = None, None, None
        self.feaDict, self.feaDict_test, self.feaDict_enroll = None, None, None
        # parameters:################################################################
        self.distrib_nb = 2048  # number of Gaussian distributions for each GMM
        self.rank_TV = 400  # 400 Rank of the total variability matrix
        self.tv_iteration = 10  # number of iterations to run
        self.plda_rk = self.rank_TV  # 400 rank of the PLDA eigenvalues matrix,if use lda or anyother operate,this value will change
        self.feature_dir = inputDir + 'fea/'  # directory where to find the features
        self.nbThread = nbThread  # Number of parallel process to run,cause machine has 32 cpus,i set it is 32
        self.feaSize = 39  # feature size that include (23mfcc+1energy)+24derived+24accelerate
        self.batchSize = 100  # this is use in  calculating stat and ivector,but plda's batch is much bigger than it(as it set to 1000)
        self.lda_rk = 400  # rank of lda
        self.trainDataDir = self.feature_dir + "train_mfcc.h5"
        self.testDataDir = self.feature_dir + "test_mfcc.h5"
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
        self.trial_key = sidekit.Key(inputDir + "trial_key_" + saveFlag + ".h5")
        self.test_ndx = sidekit.Ndx(inputDir + "test_ndx_" + saveFlag + ".h5")
        self.enroll_idmap = sidekit.IdMap(inputDir + "enroll_idmap_" + saveFlag + ".h5")
        self.test_idmap = sidekit.IdMap(inputDir + "test_idmap_" + saveFlag + ".h5")
        ###############################################################################################
        # self.test_idmap.rightids=np.asarray([i[:5]+"_"+i[-8:]  for i in list(self.test_idmap.rightids)])
        # self.enroll_idmap.rightids=np.asarray([i[:5]+"_"+i[-8:]  for i in list(self.enroll_idmap.rightids)])
        #

        ###############################################################################################

        self.ubm_TV_idmap = sidekit.IdMap(inputDir + "ubm_TV_idmap.h5")
        if trainNum != -1:
            self.ubm_TV_idmap.leftids = self.ubm_TV_idmap.leftids[:trainNum]
            self.ubm_TV_idmap.rightids = self.ubm_TV_idmap.rightids[:trainNum]
            self.ubm_TV_idmap.start = self.ubm_TV_idmap.start[:trainNum]
            self.ubm_TV_idmap.stop = self.ubm_TV_idmap.stop[:trainNum]

        self.plda_idmap = sidekit.IdMap(inputDir + "plda_idmap.h5")
        # </editor-fold>

        # feature Server ###############################################################
        self.feaServer = FeaServer(feature_filename_structure=self.feature_dir + "{}.h5")

    '''Step1: Create the universal background model from all the training speaker data'''

    @ut.timing("trainUBM")
    def trainUBM(self, data=None):
        self.ubm = Mixture()
        ##############################################################################################
        # self.feaServer.fea = self.feaServer.stack_features_parallel2(ubm_list[:5], num_thread=self.nbThread)
        # features=np.concatenate(list(self.feaServer.fea.values()),axis=0)
        # Mixture.mpi_EM_split(self.ubm,features, self.distrib_nb, save_partial=True,
        #                        output_filename=ivectorWorkDir + 'ubm/ubm')
        ##############################################################################################
        self.ubm.EM_split(features_server=self.feaServer,
                          feature_list=list(self.ubm_TV_idmap.rightids),
                          distrib_nb=self.distrib_nb,
                          num_thread=self.nbThread,
                          llk_gain=0.001,
                          save_partial=True,
                          output_file_name=ivectorWorkDir + 'ubm/ubm',
                          ceil_cov=10,
                          floor_cov=1e-2)
        # self.trainDataDir, list(self.ubm_TV_idmap.rightids), self.distrib_nb, num_thread=self.nbThread, save_partial=True,
        #               output_file_name=ivectorWorkDir + 'ubm/ubm', llk_gain=0.001,features_server=)
        ##############################################################################################
        self.ubm.write(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
        return self

    @ut.timing("trainUBM_mpi")
    def trainUBM_mpi(self):
        self.ubm = mpi_EM_split(Mixture(), self.feaServer, list(self.ubm_TV_idmap.rightids), self.distrib_nb,
                                ivectorWorkDir + 'ubm/ubm',
                                save_partial=True, ceil_cov=10, floor_cov=1e-2, num_thread=self.nbThread, logger=logger)

        return self

    '''Step2.1: Calculate the statistics from train data set needed for the iVector model.'''

    @ut.timing("calStat")
    def calStat(self, data=None):
        if self.ubm == None:
            self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
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

        with ut.Timing("calStat_background"):
            feaS = FeaServer(feature_filename_structure=self.feature_dir + "{}.h5",
                             dataset_list=["cep"])
            with ut.Timing("calStat_accumulate_stat_train"):
                self.back_stat = StatServer(self.ubm_TV_idmap, self.distrib_nb, self.feaSize)
                self.back_stat.accumulate_stat(ubm=self.ubm, feature_server=feaS,
                                               seg_indices=range(self.back_stat.segset.shape[0]),
                                               num_thread=self.nbThread)
                self.back_stat.write(ivectorWorkDir + 'stat/train_{}.h5'.format(self.distrib_nb))

        with ut.Timing("calStat_enroll"):
            feaS_e = FeaServer(feature_filename_structure=self.feature_dir + "{}.h5",
                               dataset_list=["cep"])

            self.enroll_stat = StatServer(self.enroll_idmap, self.distrib_nb, self.feaSize)
            self.enroll_stat.accumulate_stat(ubm=self.ubm, feature_server=feaS_e,
                                             seg_indices=range(self.enroll_stat.segset.shape[0]),
                                             num_thread=self.nbThread)
            self.enroll_stat.write(ivectorWorkDir + "stat/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb))

        with ut.Timing("calStat_test"):
            feaS_t = FeaServer(feature_filename_structure=self.feature_dir + "{}.h5",
                               dataset_list=["cep"])

            self.test_stat = StatServer(self.test_idmap, self.distrib_nb, self.feaSize)
            self.test_stat.accumulate_stat(ubm=self.ubm, feature_server=feaS_t,
                                           seg_indices=range(self.test_stat.segset.shape[0]), num_thread=self.nbThread)
            self.test_stat.write(ivectorWorkDir + "stat/test_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        return self

    '''Step2.2: Learn the total variability subspace from all the train speaker data.'''

    @ut.timing("learnTV")
    def learnTV_deprecate(self, data=None):
        if self.ubm == None:
            self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
        if self.back_stat == None:
            with ut.Timing("learnTV_StatServer"):
                self.back_stat = StatServer(ivectorWorkDir + 'stat/train_{}.h5'.format(
                    self.distrib_nb))  # this is super class,but not child class
        # cause Tmatrix compute so complex,i only use 10000 utterances to train it
        stat_bak = copy.deepcopy(self.back_stat)
        # self.back_stat.partition(tvNum)
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

        # sidekit.sidekit_io.write_tv_hdf5((tv, tv_mean, tv_sigma),
        #                                  ivectorWorkDir + "Tmatrix/T_{}".format(self.distrib_nb))
        self.back_stat = stat_bak
        return self

    @ut.timing("learnTV")
    def learnTV_mpi(self, data=None):
        if self.ubm == None:
            self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
        stat_server_file_name = ivectorWorkDir + 'stat/train_{}.h5'.format(self.distrib_nb)
        output_file_name = ivectorWorkDir + "Tmatrix/T_{}".format(self.distrib_nb)
        mpi_learnTV(stat_server_file_name=stat_server_file_name,
                    ubm=self.ubm,
                    tv_rank=self.rank_TV,
                    nb_iter=self.tv_iteration,
                    min_div=True,
                    tv_init=None,
                    save_init=False,
                    output_file_name=output_file_name, logger=logger)
        return self

    @ut.timing("learnTV")
    def learnTV(self, data=None):
        if self.ubm == None:
            self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
        stat_server_file_name = ivectorWorkDir + 'stat/train_{}.h5'.format(self.distrib_nb)
        output_file_name = ivectorWorkDir + "Tmatrix/T_{}".format(self.distrib_nb)

        fa = FA()
        fa.total_variability(stat_server_filename=stat_server_file_name,
                             ubm=self.ubm,
                             tv_rank=self.rank_TV,
                             nb_iter=self.tv_iteration,
                             min_div=True,
                             tv_init=None,
                             batch_size=self.batchSize * 3,
                             save_init=False,
                             output_file_name=output_file_name,
                             num_thread=self.nbThread)
        self.Tmatrix = fa
        return self

    '''Step2.3:Now compute the development ivectors of train data set for each speaker and channel.  The result is size tvDim x nSpeakers x nChannels.'''

    @ut.timing("extractIV")
    def extractIV(self, data=None):

        if self.ubm == None:
            self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
        if self.Tmatrix == None:
            self.Tmatrix = FactorAnalyser(ivectorWorkDir + 'Tmatrix/T_{}.h5'.format(self.distrib_nb))

        with ut.Timing("background_iv"):
            stat_server_file_name = ivectorWorkDir + 'stat/train_{}.h5'.format(self.distrib_nb)
            self.train_iv = self.Tmatrix.extract_ivectors(self.ubm, stat_server_file_name, num_thread=self.nbThread)
            self.train_iv.write(ivectorWorkDir + "iv/train_{}.h5".format(self.distrib_nb))
        with ut.Timing("enroll_test_iv"):
            stat_server_file_name = ivectorWorkDir + "stat/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb)
            self.enroll_iv = self.Tmatrix.extract_ivectors(self.ubm, stat_server_file_name, num_thread=self.nbThread)
            self.enroll_iv.write(ivectorWorkDir + "iv/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            stat_server_file_name = ivectorWorkDir + "stat/test_" + saveFlag + "_{}.h5".format(self.distrib_nb)
            self.test_iv = self.Tmatrix.extract_ivectors(self.ubm, stat_server_file_name, num_thread=self.nbThread)
            self.test_iv.write(ivectorWorkDir + "iv/test_" + saveFlag + "_{}.h5".format(self.distrib_nb))

        return self

    @ut.timing("extractIV")
    def extractIV_deprecate(self, data=None):
        #####################################################
        # just extract which i want to use in future,e.g:enroll,test ,train portion in sre04-08,
        # so that i would not extract ivector on switchnboard,cause i just use sre04-08 to train lda and plda
        if self.Tmatrix == None:
            tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5(
                ivectorWorkDir + "Tmatrix/T_{}.h5".format(self.distrib_nb))
        else:
            tv, tv_mean, tv_sigma = self.Tmatrix
            # free space
            del self.Tmatrix
        with ut.Timing("background_iv"):
            ##########################################################################################
            # cause that plda train data is not the same as before,so there must be read
            # self.back_stat = StatServer.read_subset(ivectorWorkDir + "stat/train_{}.h5".format(self.distrib_nb),
            #                                         self.plda_idmap)
            ##########################################################################################
            # but i have only use first 20000 data,so there has no swb data,i.e. this code will except error,
            # i revise it as follow so that it could run by comment this code line,in this case,it will
            # use all 20000 data to train plda(swb data has around 20000 data in fact.)
            if self.back_stat == None:
                self.back_stat = StatServer(
                    ivectorWorkDir + "stat/train_{}.h5".format(self.distrib_nb))
            ##########################################################################################
            train_iv = \
                self.back_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=self.batchSize,
                                               num_thread=self.nbThread)[0]
            train_iv.write(ivectorWorkDir + "iv/train_{}.h5".format(self.distrib_nb))

        with ut.Timing("enroll_test_iv"):
            if self.enroll_stat == None:
                self.enroll_stat = StatServer(
                    ivectorWorkDir + "stat/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb))

            enroll_iv = \
                self.enroll_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=self.batchSize,
                                                 num_thread=self.nbThread)[0]
            enroll_iv.write(ivectorWorkDir + "iv/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            if self.test_stat == None:
                self.test_stat = StatServer(
                    ivectorWorkDir + "stat/test_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            test_iv = \
                self.test_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=self.batchSize,
                                               num_thread=self.nbThread)[0]
            test_iv.write(ivectorWorkDir + "iv/test_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            del self.enroll_stat
            del self.test_stat

        return self

    @ut.timing("extractIV")
    def extractIV_mpi(self, data=None):
        # just extract which i want to use in future,e.g:enroll,test ,train portion in sre04-08,
        # so that i would not extract ivector on switchnboard,cause i just use sre04-08 to train lda and plda
        if self.Tmatrix == None:
            self.Tmatrix = FactorAnalyser(ivectorWorkDir + 'Tmatrix/T_{}.h5'.format(self.distrib_nb))

        if self.ubm == None:
            self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))

        with ut.Timing("background"):
            ##########################################################################################
            # cause that plda train data is not the same as before,so there must be read
            # self.back_stat = StatServer.read_subset(ivectorWorkDir + "stat/train_{}.h5".format(self.distrib_nb),
            #                                         self.plda_idmap)
            ##########################################################################################
            # but i have only use first 20000 data,so there has no swb data,i.e. this code will except error,
            # i revise it as follow so that it could run by comment this code line,in this case,it will
            # use all 20000 data to train plda(swb data has around 20000 data in fact.)

            statserver_file_name = ivectorWorkDir + "stat/train_{}.h5".format(self.distrib_nb)
            mpi_extractIV(self.Tmatrix, statserver_file_name, self.ubm,
                          ivectorWorkDir + "iv/train_{}.h5".format(self.distrib_nb))
            ##########################################################################################
            # non mpi mode
            # if self.back_stat == None:
            #     self.back_stat = StatServer(statserver_file_name)
            # self.Tmatrix.extract_ivectors(self.ubm,self.back_stat,'',self.batchSize*3,False,self.nbThread)
            ##########################################################################################
        with ut.Timing("enroll_test"):
            statserver_file_name = ivectorWorkDir + "stat/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb)
            mpi_extractIV(self.Tmatrix, statserver_file_name, self.ubm,
                          ivectorWorkDir + "iv/enroll_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            # non mpi mode

            # if self.enroll_stat == None:
            #     self.enroll_stat = StatServer(statserver_file_name)
            # self.Tmatrix.extract_ivectors(self.ubm,self.enroll_stat,'',self.batchSize*3,False,self.nbThread)
            ##########################################################################################
            statserver_file_name = ivectorWorkDir + "stat/test_" + saveFlag + "_{}.h5".format(self.distrib_nb)
            mpi_extractIV(self.Tmatrix, statserver_file_name, self.ubm,
                          ivectorWorkDir + "iv/test_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            # non mpi mode

            # if self.test_stat == None:
            #     self.test_stat = StatServer(statserver_file_name)
            # self.Tmatrix.extract_ivectors(self.ubm,self.test_stat,'',self.batchSize*3,False,self.nbThread)
            ##########################################################################################
        return self


        ##############################################################################################################
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
        with ut.Timing("score/cos_" + saveFlag):
            train_iv_wccn, enroll_iv_wccn, test_iv_wccn = self.deepcopy(self.train_iv, self.enroll_iv, self.test_iv)

            scores_cos = sidekit.iv_scoring.cosine_scoring(enroll_iv_wccn, test_iv_wccn, self.test_ndx, wccn=None)
            scores_cos.write(ivectorWorkDir + "score/cos_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###########################################################################
        with ut.Timing("score/cos_wccn_" + saveFlag):
            wccn = train_iv_wccn.get_wccn_choleski_stat1()
            scores_cos_wccn = sidekit.iv_scoring.cosine_scoring(enroll_iv_wccn, test_iv_wccn, self.test_ndx, wccn=wccn)
            scores_cos_wccn.write(ivectorWorkDir + "score/cos_wccn_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###########################################################################
        with ut.Timing("score/cos_lda_" + saveFlag):
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
        with ut.Timing("score/cos_lda_wccn_" + saveFlag):
            wccn = self.train_iv_lda.get_wccn_choleski_stat1()
            scores_cos_lda_wcnn = sidekit.iv_scoring.cosine_scoring(self.enroll_iv_lda, self.test_iv_lda, self.test_ndx,
                                                                    wccn=wccn)
            scores_cos_lda_wcnn.write(
                ivectorWorkDir + "score/cos_lda_wccn_" + saveFlag + "_{}.h5".format(self.distrib_nb))
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
        with ut.Timing("score/2covar_" + saveFlag):
            W = train_iv1.get_within_covariance_stat1()
            B = train_iv1.get_between_covariance_stat1()
            scores_2cov = sidekit.iv_scoring.two_covariance_scoring(enroll_iv1, test_iv1, self.test_ndx, W, B)
            scores_2cov.write(ivectorWorkDir + "score/2covar_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ########################################################
        with ut.Timing("score/2covar_sphernorm_" + saveFlag):
            meanSN, CovSN = self.train_iv.estimate_spectral_norm_stat1(1, "sphNorm")

            self.train_iv.spectral_norm_stat1(meanSN[:1], CovSN[:1])
            self.enroll_iv.spectral_norm_stat1(meanSN[:1], CovSN[:1])
            self.test_iv.spectral_norm_stat1(meanSN[:1], CovSN[:1])
            W1 = self.train_iv.get_within_covariance_stat1()
            B1 = self.train_iv.get_between_covariance_stat1()
            scores_2cov_sn1 = sidekit.iv_scoring.two_covariance_scoring(self.enroll_iv, self.test_iv, self.test_ndx, W1,
                                                                        B1)
            scores_2cov_sn1.write(
                ivectorWorkDir + "score/2covar_sphernorm_" + saveFlag + "_{}.h5".format(self.distrib_nb))
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

        with ut.Timing("score/plda_sphernorm_" + saveFlag):
            ##########################################################################################################
            # output_file_name = ivectorWorkDir + "plda/plda_sphernorm_" + saveFlag + "_{}".format(self.distrib_nb)
            # if self.ubm == None:
            #     self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
            # comm=MPI.COMM_WORLD
            #
            # if comm.rank==0:
            #     self.train_iv.write(ivectorWorkDir+"tmp/train_iv.h5")
            # comm.Barrier()
            # stat_server_file_name = ivectorWorkDir+"tmp/train_iv.h5"
            # fa=mpi_learnTV(stat_server_file_name=stat_server_file_name,
            #                 ubm=self.ubm,
            #                 tv_rank=self.plda_rk,
            #                 nb_iter=self.tv_iteration,
            #                 min_div=True,
            #                 tv_init=None,
            #                 save_init=False,
            #                 output_file_name=output_file_name,logger=logger)
            # plda_mean, plda_F,plda_G,plda_H,plda_Sigma=fa.mean,fa.F,fa.G,fa.H,fa.Sigma
            ##########################################################################################################
            # non mpi
            output_file_name = ivectorWorkDir + "plda/plda_sphernorm_" + saveFlag + "_{}".format(self.distrib_nb)
            fa = FactorAnalyser()
            fa.plda(self.train_iv, self.plda_rk, self.tv_iteration, 1, output_file_name, False)  # self.tv_iteration
            plda_mean, plda_F, plda_G, plda_H, plda_Sigma = fa.mean, fa.F, fa.G, fa.H, fa.Sigma
            plda_G = np.zeros(
                plda_F.shape[0])  # cause that module has a assertion ,there must has G,so that to avoid error.
            # plda_mean, plda_F, plda_G, plda_H, plda_Sigma = self.train_iv.factor_analysis(rank_f=self.plda_rk, rank_g=0,
            #                                                                               rank_h=None,
            #                                                                               re_estimate_residual=True,
            #                                                                               it_nb=(10, 0, 0), min_div=True,
            #                                                                               ubm=None,
            #                                                                               batch_size=self.batchSize * 10,
            #                                                                               num_thread=self.nbThread)
            #
            # sidekit.sidekit_io.write_plda_hdf5((plda_mean, plda_F, plda_G, plda_Sigma),output_file_name)
            ##########################################################################################################


            scores_plda = sidekit.iv_scoring.PLDA_scoring(self.enroll_iv, self.test_iv, self.test_ndx, plda_mean,
                                                          plda_F,
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
        with ut.Timing("score/plda_lda_" + saveFlag):
            ##########################################################################################################
            # output_file_name = ivectorWorkDir + "plda/plda_lda_" + saveFlag + "_{}".format(self.distrib_nb)
            # if self.ubm == None:
            #     self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
            # stat_server_file_name = ivectorWorkDir + "lda/train_{}.h5".format(self.distrib_nb)
            # fa = mpi_learnTV(stat_server_file_name=stat_server_file_name,
            #                  ubm=self.ubm,
            #                  tv_rank=self.lda_rk,
            #                  nb_iter=self.tv_iteration,
            #                  min_div=True,
            #                  tv_init=None,
            #                  save_init=False,
            #                  output_file_name=output_file_name, logger=logger)
            # plda_mean, plda_F, plda_G, plda_H, plda_Sigma = fa.mean, fa.F, fa.G, fa.H, fa.Sigma
            ##########################################################################################################
            # non mpi
            output_file_name = ivectorWorkDir + "plda/plda_lda_" + saveFlag + "_{}".format(self.distrib_nb)
            fa = FactorAnalyser()
            fa.plda(train_iv1, self.lda_rk, self.tv_iteration, 1, output_file_name, False)
            plda_mean, plda_F, plda_G, plda_H, plda_Sigma = fa.mean, fa.F, fa.G, fa.H, fa.Sigma
            plda_G = np.zeros(
                plda_F.shape[0])  # cause that module has a assertion ,there must has G,so that to avoid error.

            # plda_mean, plda_F, plda_G, plda_H, plda_Sigma = train_iv1.factor_analysis(rank_f=self.lda_rk, rank_g=0,
            #                                                                           rank_h=None,
            #                                                                           re_estimate_residual=True,
            #                                                                           it_nb=(10, 0, 0), min_div=True,
            #                                                                           ubm=None,
            #                                                                           batch_size=self.batchSize * 10,
            #                                                                           num_thread=self.nbThread)
            # sidekit.sidekit_io.write_plda_hdf5((plda_mean, plda_F, plda_G, plda_Sigma),
            #                                    ivectorWorkDir + "plda/plda_lda_" + saveFlag + "_{}.h5".format(
            #                                        self.distrib_nb))
            ##########################################################################################################

            scores_plda = sidekit.iv_scoring.PLDA_scoring(enroll_iv1, test_iv1, self.test_ndx, plda_mean, plda_F,
                                                          plda_G,
                                                          plda_Sigma, full_model=False)
            scores_plda.write(ivectorWorkDir + "score/plda_lda_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###############################################################################
        train_wccn, enroll_wccn, test_wccn = self.deepcopy(self.train_iv_lda, self.enroll_iv_lda, self.test_iv_lda)
        with ut.Timing("score/plda_lda_wccn_" + saveFlag):
            wccn = train_wccn.get_wccn_choleski_stat1()
            train_wccn.rotate_stat1(wccn)
            enroll_wccn.rotate_stat1(wccn)
            test_wccn.rotate_stat1(wccn)

            train_wccn1, enroll_wccn1, test_wccn1 = self.deepcopy(train_wccn, enroll_wccn, test_wccn)

            train_wccn.write(ivectorWorkDir + "lda/train_wccn_{}.h5".format(self.distrib_nb))
            enroll_wccn.write(ivectorWorkDir + "lda/enroll_wccn_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            test_wccn.write(ivectorWorkDir + "lda/test_wccn_" + saveFlag + "_{}.h5".format(self.distrib_nb))
            ##########################################################################################################
            # output_file_name = ivectorWorkDir + "plda/plda_lda_wccn_" + saveFlag + "_{}".format(self.distrib_nb)
            # if self.ubm == None:
            #     self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
            # stat_server_file_name = ivectorWorkDir + "lda/train_wccn_{}.h5".format(self.distrib_nb)
            # fa = mpi_learnTV(stat_server_file_name=stat_server_file_name,
            #                  ubm=self.ubm,
            #                  tv_rank=self.lda_rk,
            #                  nb_iter=self.tv_iteration,
            #                  min_div=True,
            #                  tv_init=None,
            #                  save_init=False,
            #                  output_file_name=output_file_name, logger=logger)
            # plda_mean, plda_F, plda_G, plda_H, plda_Sigma = fa.mean, fa.F, fa.G, fa.H, fa.Sigma
            ##########################################################################################################
            # non mpi
            output_file_name = ivectorWorkDir + "plda/plda_lda_wccn_" + saveFlag + "_{}".format(self.distrib_nb)
            fa = FactorAnalyser()
            fa.plda(train_wccn, self.lda_rk, self.tv_iteration, 1, output_file_name, False)
            plda_mean, plda_F, plda_G, plda_H, plda_Sigma = fa.mean, fa.F, fa.G, fa.H, fa.Sigma
            plda_G = np.zeros(
                plda_F.shape[0])  # cause that module has a assertion ,there must has G,so that to avoid error.

            # plda_mean, plda_F, plda_G, plda_H, plda_Sigma = train_wccn.factor_analysis(rank_f=self.lda_rk, rank_g=0,
            #                                                                            rank_h=None,
            #                                                                            re_estimate_residual=True,
            #                                                                            it_nb=(10, 0, 0), min_div=True,
            #                                                                            ubm=None,
            #                                                                            batch_size=self.batchSize * 10,
            #                                                                            num_thread=self.nbThread)
            # sidekit.sidekit_io.write_plda_hdf5((plda_mean, plda_F, plda_G, plda_Sigma),
            #                                    ivectorWorkDir + "plda/plda_lda_wccn_" + saveFlag + "_{}.h5".format(
            #                                        self.distrib_nb))
            ##########################################################################################################
            scores_plda = sidekit.iv_scoring.PLDA_scoring(enroll_wccn, test_wccn, self.test_ndx, plda_mean, plda_F,
                                                          plda_G,
                                                          plda_Sigma, full_model=False)
            scores_plda.write(ivectorWorkDir + "score/plda_lda_wccn_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###############################################################################
        with ut.Timing("score/plda_lda_wccn_sphernorm_" + saveFlag):
            meanSN, CovSN = train_wccn1.estimate_spectral_norm_stat1(1, "sphNorm")
            train_wccn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
            enroll_wccn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
            test_wccn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
            ##########################################################################################################
            # output_file_name = ivectorWorkDir + "plda/plda_lda_wccn__sphernorm_" + saveFlag + "_{}".format(self.distrib_nb)
            # if self.ubm == None:
            #     self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
            # comm = MPI.COMM_WORLD
            #
            # if comm.rank == 0:
            #     train_wccn1.write(ivectorWorkDir+"tmp/train_plda_lda_wccn_sphernorm.h5")
            # comm.Barrier()
            #
            # stat_server_file_name = ivectorWorkDir+"tmp/train_plda_lda_wccn_sphernorm.h5"
            # fa = mpi_learnTV(stat_server_file_name=stat_server_file_name,
            #                  ubm=self.ubm,
            #                  tv_rank=self.lda_rk,
            #                  nb_iter=self.tv_iteration,
            #                  min_div=True,
            #                  tv_init=None,
            #                  save_init=False,
            #                  output_file_name=output_file_name, logger=logger)
            # plda_mean, plda_F, plda_G, plda_H, plda_Sigma = fa.mean, fa.F, fa.G, fa.H, fa.Sigma
            ##########################################################################################################
            # non mpi
            output_file_name = ivectorWorkDir + "plda/plda_lda_wccn__sphernorm_" + saveFlag + "_{}".format(
                self.distrib_nb)
            fa = FactorAnalyser()
            fa.plda(train_wccn1, self.lda_rk, self.tv_iteration, 1, output_file_name, False)
            plda_mean, plda_F, plda_G, plda_H, plda_Sigma = fa.mean, fa.F, fa.G, fa.H, fa.Sigma
            plda_G = np.zeros(
                plda_F.shape[0])  # cause that module has a assertion ,there must has G,so that to avoid error.

            # plda_mean, plda_F, plda_G, plda_H, plda_Sigma = train_wccn1.factor_analysis(rank_f=self.lda_rk, rank_g=0,
            #                                                                             rank_h=None,
            #                                                                             re_estimate_residual=True,
            #                                                                             it_nb=(10, 0, 0), min_div=True,
            #                                                                             ubm=None,
            #                                                                             batch_size=self.batchSize * 10,
            #                                                                             num_thread=self.nbThread)
            # sidekit.sidekit_io.write_plda_hdf5((plda_mean, plda_F, plda_G, plda_Sigma),
            #                                    ivectorWorkDir + "plda/plda_lda_wccn__sphernorm_" + saveFlag + "_{}.h5".format(
            #                                        self.distrib_nb))
            ##########################################################################################################
            scores_plda = sidekit.iv_scoring.PLDA_scoring(enroll_wccn1, test_wccn1, self.test_ndx, plda_mean, plda_F,
                                                          plda_G,
                                                          plda_Sigma, full_model=False)
            scores_plda.write(
                ivectorWorkDir + "score/plda_lda_wccn_sphernorm_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        ###############################################################################
        with ut.Timing("score/plda_lda_sphernorm_" + saveFlag):
            meanSN, CovSN = self.train_iv_lda.estimate_spectral_norm_stat1(1, "sphNorm")
            self.train_iv_lda.spectral_norm_stat1(meanSN[:1], CovSN[:1])
            self.enroll_iv_lda.spectral_norm_stat1(meanSN[:1], CovSN[:1])
            self.test_iv_lda.spectral_norm_stat1(meanSN[:1], CovSN[:1])
            ##########################################################################################################
            # output_file_name = ivectorWorkDir + "plda/plda_lda_sphernorm_" + saveFlag + "_{}".format(self.distrib_nb)
            # if self.ubm == None:
            #     self.ubm = Mixture(ivectorWorkDir + 'ubm/ubm_{}.h5'.format(self.distrib_nb))
            #
            # comm = MPI.COMM_WORLD
            #
            # if comm.rank == 0:
            #     self.train_iv_lda.write(ivectorWorkDir + "tmp/train_plda_lda_sphernorm.h5")
            # comm.Barrier()
            #
            # stat_server_file_name = ivectorWorkDir + "tmp/train_plda_lda_sphernorm.h5"
            # fa = mpi_learnTV(stat_server_file_name=stat_server_file_name,
            #                  ubm=self.ubm,
            #                  tv_rank=self.lda_rk,
            #                  nb_iter=self.tv_iteration,
            #                  min_div=True,
            #                  tv_init=None,
            #                  save_init=False,
            #                  output_file_name=output_file_name, logger=logger)
            # plda_mean, plda_F, plda_G, plda_H, plda_Sigma = fa.mean, fa.F, fa.G, fa.H, fa.Sigma
            ##########################################################################################################
            # non mpi
            output_file_name = ivectorWorkDir + "plda/plda_lda_sphernorm_" + saveFlag + "_{}".format(self.distrib_nb)
            fa = FactorAnalyser()
            fa.plda(self.train_iv_lda, self.lda_rk, self.tv_iteration, 1, output_file_name, False)
            plda_mean, plda_F, plda_G, plda_H, plda_Sigma = fa.mean, fa.F, fa.G, fa.H, fa.Sigma
            plda_G = np.zeros(
                plda_F.shape[0])  # cause that module has a assertion ,there must has G,so that to avoid error.

            # plda_mean, plda_F, plda_G, plda_H, plda_Sigma = self.train_iv_lda.factor_analysis(rank_f=self.lda_rk, rank_g=0,
            #                                                                                   rank_h=None,
            #                                                                                   re_estimate_residual=True,
            #                                                                                   it_nb=(10, 0, 0),
            #                                                                                   min_div=True,
            #                                                                                   ubm=None,
            #                                                                                   batch_size=self.batchSize * 10,
            #                                                                                   num_thread=self.nbThread)
            # sidekit.sidekit_io.write_plda_hdf5((plda_mean, plda_F, plda_G, plda_Sigma),
            #                                    ivectorWorkDir + "plda/plda_lda_sphernorm_" + saveFlag + "_{}.h5".format(
            #                                        self.distrib_nb))
            ##########################################################################################################
            scores_plda = sidekit.iv_scoring.PLDA_scoring(self.enroll_iv_lda, self.test_iv_lda, self.test_ndx,
                                                          plda_mean,
                                                          plda_F, plda_G,
                                                          plda_Sigma, full_model=False)
            scores_plda.write(
                ivectorWorkDir + "score/plda_lda_sphernorm_" + saveFlag + "_{}.h5".format(self.distrib_nb))
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
        dp = sidekit.DetPlot(window_style="sre10",
                             plot_title=ivectorWorkDir + "result/IVector_SRE_2010" + str(self.rank_TV) + "_" + str(
                                 self.lda_rk))
        dp.create_figure()

        graphData = ["cos", "cos_wccn", "cos_lda", "cos_lda_wccn", "madis", "2covar", "2covar_sphernorm",
                     "plda_sphernorm", "plda_lda", "plda_lda_wccn", "plda_lda_wccn_sphernorm", "plda_lda_sphernorm"]
        for j, i in enumerate(graphData):
            tmp = ivectorWorkDir + "score/" + i + "_" + saveFlag + "_{}.h5".format(self.distrib_nb)
            if os.path.exists(tmp):
                score = sidekit.Scores(tmp)
                dp.set_system_from_scores(score, self.trial_key, sys_name=i)
                dp.plot_rocch_det(j, target_prior=prior)

        # dp.plot_DR30_both(idx=0)
        dp.plot_mindcf_point(prior, idx=0)

    def readiv(self, fileList: list = None) -> (sidekit.StatServer, sidekit.StatServer, sidekit.StatServer):
        train_iv = sidekit.StatServer.read(ivectorWorkDir + fileList[0] + "_{}.h5".format(self.distrib_nb))
        enroll_iv = sidekit.StatServer.read(
            ivectorWorkDir + fileList[1] + "_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        test_iv = sidekit.StatServer.read(
            ivectorWorkDir + fileList[2] + "_" + saveFlag + "_{}.h5".format(self.distrib_nb))
        enroll_iv, test_iv = self.fixNameProplem(enroll_iv, test_iv)
        return train_iv, enroll_iv, test_iv

    def fixNameProplem(self, enroll_iv, test_iv):
        # enroll_iv.segset=np.asarray([i[:5]+"_"+i[-8:] for i in list(enroll_iv.segset)])
        # enroll_iv.modelset = enroll_iv.segset
        # test_iv.segset = np.asarray([i[:5] + "_" + i[-8:] for i in list(test_iv.segset)])
        return enroll_iv, test_iv

    def deepcopy(self, train_iv, enroll_iv, test_iv):
        train_iv_wccn = copy.deepcopy(train_iv)
        enroll_iv_wccn = copy.deepcopy(enroll_iv)
        test_iv_wccn = copy.deepcopy(test_iv)
        return train_iv_wccn, enroll_iv_wccn, test_iv_wccn

    def test(self):
        s = self.feaServer.load("sre10/10sec/1/lfwsgTB1", input_feature_filename=None, label=None, start=None,
                                stop=None)
        pass

    # note that the rights problem,be sure that you have rights to write it
    def createDir(self):

        isCreate = ut.creatDirIfNotExists(ivectorWorkDir + "tmp")
        ut.creatDirIfNotExists(ivectorWorkDir + "lda")
        ut.creatDirIfNotExists(ivectorWorkDir + "stat")
        ut.creatDirIfNotExists(ivectorWorkDir + "result")
        ut.creatDirIfNotExists(ivectorWorkDir + "plda")
        ut.creatDirIfNotExists(ivectorWorkDir + "Tmatrix")
        ut.creatDirIfNotExists(ivectorWorkDir + "iv")
        ut.creatDirIfNotExists(ivectorWorkDir + "ubm")
        ut.creatDirIfNotExists(ivectorWorkDir + "score")
        if isCreate:
            print("directory may has no rights for you,please add rights to yourself,manully")
            exit(1)

    def selectDataForPlda(self):
        '''
        just use SWB data to train PLDA,this method is modify data on disk,,So other code needn't rewrite.
        note that revising is permanent,so if want use origin data,i should rename data on disk.
        '''
        if self.train_iv == None:
            train_iv = sidekit.StatServer.read(ivectorWorkDir + "iv/train_{}.h5".format(self.distrib_nb))
        ind = [i for i, j in enumerate(train_iv.segset) if re.match("^((phase)|(cellular))/.*", j)]
        if len(ind) == 0:
            print("\n\n\n--------------------Does not do selectDataForPlda()------------------\n\n\n")
            return self
        newSer = sidekit.StatServer()
        modelset, segset, stat1 = [], [], []
        for i in ind:
            segset.append(train_iv.segset[i])
            modelset.append(train_iv.modelset[i])
            stat1.append(train_iv.stat1[i][None])
        newSer.segset = np.array(segset)
        newSer.modelset = np.array(modelset)
        newSer.stat1 = np.concatenate(stat1)
        newSer.start = np.array([None] * len(ind))
        newSer.stop = np.array([None] * len(ind))
        newSer.stat0 = np.array([[1.0]] * len(ind))
        # os.system("mv "+ivectorWorkDir + "iv/train_{}.h5".format(self.distrib_nb)+" "+ivectorWorkDir + "iv/train_{}.h5_all".format(self.distrib_nb))
        # newSer.write(ivectorWorkDir + "iv/train_{}.h5".format(self.distrib_nb))
        self.train_iv = newSer
        return self


def main():
    ut.creatDirIfNotExists(ivectorWorkDir)

    # fea=FeaServer(dataset_list=["ssss"])
    # fea.stack_features_parallel(["ss","qq","cc"],num_thread=3)
    # if True:
    #     return 0

    # DataInteger(False)
    # DataInteger.readData()
    logger.info("=================================================================================================")
    # te=Mixture('/home/jyh/D/jyh/data/fea/ivector_test/ubm/ubm_128.h5')

    # Tmatrix = FactorAnalyser(ivectorWorkDir + 'Tmatrix/T_{}.h5'.format(64))
    # Tmatrix1 = FactorAnalyser(ivectorWorkDir + 'Tmatrix/T_{}.h5_'.format(64))

    # DataInteger()
    iv = IV(48)
    iv.selectDataForPlda()
    # iv.visualization()
    # iv.calStat()
    statserver_file_name = ivectorWorkDir + "stat/train_{}.h5".format(1024)
    tr = sidekit.StatServer(statserver_file_name)
    statserver_file_name = ivectorWorkDir + "stat/test_" + saveFlag + "_{}.h5".format(64)
    te = sidekit.StatServer(statserver_file_name)
    statserver_file_name = ivectorWorkDir + "stat/test_" + saveFlag + "_{}.h5_".format(64)
    te1 = sidekit.StatServer(statserver_file_name)
    # iv.extractIV()
    # #iv.PLDA_Score().visualization()
    # .calStat().learnTV_mpi().extractIV_mpi()
    iv.two_covariance_Score().selectDataForPlda().PLDA_Score().PLDA_LDA_Score().visualization()
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
    DataInteger()

    # main()
