#!/usr/bin/env python
# -*- coding:utf-8 -*-
import socket
from mpi4py import MPI
import sidekit
import jyh.Utils as ut
import os

os.environ["SIDEKIT"] = "theano=false,theano_config=cpu,libsvm=false,mpi=false"


# class Fea(sidekit.FeaturesExtractor):
#     def __init__(self):
#         self.extra = None
#         super(Fea, self).__init__(audio_filename_structure="/home/jyh/share/{}.sph",
#                                   feature_filename_structure="/home/jyh/share/{}.h51",  # the 2nd is channel ,it is added by me
#                                   sampling_frequency=8000,
#                                   lower_frequency=200,
#                                   higher_frequency=3800,
#                                   filter_bank="log",
#                                   filter_bank_size=40,
#                                   window_size=0.025,
#                                   shift=0.01,
#                                   ceps_number=12,
#                                   vad="energy",
#                                   pre_emphasis=0.97,
#                                   save_param=["vad", "energy", "cep", "fb"],
#                                   keep_all_features=True)
#
#         self.feaServer = sidekit.FeaturesServer(features_extractor=self,
#                                                 feature_filename_structure=None,
#                                                 sources=None,
#                                                 dataset_list=None,  # ["energy","cep","fb"],
#                                                 feat_norm="cmvn_sliding",  # cms cmvn stg cmvn_sliding cms_sliding
#                                                 delta=True,
#                                                 double_delta=True,
#                                                 rasta=True,
#                                                 keep_all_features=True,
#                                                 # mask="[0-12]",
#                                                 # global_cmvn=None,
#                                                 # dct_pca=False,
#                                                 # dct_pca_config=None,
#                                                 # sdc=False,
#                                                 # sdc_config=None,
#                                                 # delta_filter=None,
#                                                 # context=None,
#                                                 # traps_dct_nb=None,
#                                                 )
#
#
#         # s=server.load('sre04/xgvxTA', input_feature_filename=None, label=None, start=None,stop = None)
#

def main():
    test()


def test1():
    pass
    # feaExtract=sidekit.FeaturesExtractor(audio_filename_structure="/home/jyh/share/{}.sph",
    #                                       feature_filename_structure="/home/jyh/share/{}.h51",
    #                                       # the 2nd is channel ,it is added by me
    #                                       sampling_frequency=8000,
    #                                       lower_frequency=200,
    #                                       higher_frequency=3800,
    #                                       filter_bank="log",
    #                                       filter_bank_size=40,
    #                                       window_size=0.025,
    #                                       shift=0.01,
    #                                       ceps_number=12,
    #                                       vad="energy",
    #                                       pre_emphasis=0.97,
    #                                       save_param=["vad", "energy", "cep", "fb"],
    #                                       keep_all_features=True)
    #
    # feaServer = sidekit.FeaturesServer(features_extractor=feaExtract,
    #                                         feature_filename_structure=None,
    #                                         sources=None,
    #                                         dataset_list=None,  # ["energy","cep","fb"],
    #                                         feat_norm="cmvn_sliding",  # cms cmvn stg cmvn_sliding cms_sliding
    #                                         delta=True,
    #                                         double_delta=True,
    #                                         rasta=True,
    #                                         keep_all_features=True,
    #                                         # mask="[0-12]",
    #                                         # global_cmvn=None,
    #                                         # dct_pca=False,
    #                                         # dct_pca_config=None,
    #                                         # sdc=False,
    #                                         # sdc_config=None,
    #                                         # delta_filter=None,
    #                                         # context=None,
    #                                         # traps_dct_nb=None,
    #                                         )
    # feaServer.load("laaem",0)
    #
    # s = fea.extract(show="laaem",channel=0)
    # t = fea.extract(show="laagc",channel=0)
    # s1 = fea.extract(show="laaem",channel=1)
    # t1 = fea.extract(show="laagc",channel=1)
    # pass


def test():
    comm = MPI.COMM_WORLD
    print("This is Process: {} over {}".format(comm.rank, comm.size))
    if comm.rank == 0:
        print("I'm process 0")
    name = socket.gethostname()
    print("hostname:" + name)


if __name__ == '__main__':
    main()
