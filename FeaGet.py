#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

from sidekit import vad_energy

os.environ["SIDEKIT"] = "theano=false,theano_config=cpu,libsvm=false,mpi=false"
from sidekit.frontend.io import read_audio, write_hdf5
from sidekit.frontend.features import mfcc, plp
import h5py
from multiprocessing import Pool
import re
from sidekit.features_server import FeaturesServer
import sidekit
import pandas as pd
from PrepareData import IdMapConstructor as idc
import numpy as np
import jyh.Utils as ut
import pickle as pk
import globalVar as glb

root = glb.get_root()


# use my multiprocess
class Fea(sidekit.FeaturesExtractor):
    def __init__(self):
        self.extra = None
        super(Fea, self).__init__(audio_filename_structure=None,
                                  feature_filename_structure=None,  # the 2nd is channel ,it is added by me
                                  sampling_frequency=8000,
                                  lower_frequency=200,
                                  higher_frequency=3800,
                                  filter_bank="log",
                                  filter_bank_size=40,
                                  window_size=0.025,
                                  shift=0.01,
                                  ceps_number=12,
                                  vad="energy",
                                  pre_emphasis=0.97,
                                  save_param=["vad", "energy", "cep", "fb"],
                                  keep_all_features=True)

        self.feaServer = sidekit.FeaturesServer(features_extractor=None,
                                                feature_filename_structure=None,
                                                sources=None,
                                                dataset_list=None,  # ["energy","cep","fb"],
                                                feat_norm=None,
                                                # cms cmvn stg cmvn_sliding cms_sliding,there do not do it ,we can do it in next step
                                                delta=True,
                                                double_delta=True,
                                                rasta=True,
                                                keep_all_features=False,
                                                # mask="[0-12]",
                                                # global_cmvn=None,
                                                # dct_pca=False,
                                                # dct_pca_config=None,
                                                # sdc=False,
                                                # sdc_config=None,
                                                # delta_filter=None,
                                                # context=None,
                                                # traps_dct_nb=None,
                                                )


        # s=server.load('sre04/xgvxTA', input_feature_filename=None, label=None, start=None,stop = None)

    @ut.timing('getFea')
    def feaExtract(self, num_thread, year: list = None, extract: bool = True):
        if year is None:
            year = [4, 5, 6, 8, 'cellular', 'phase']  # 10 ,4,5,6,8,'cellular','phase'

        idmapAll = idc.readAllSreIdmap()
        for i in year:

            y = idmapAll[str(i)]
            if len(y) == 4:
                name = list(np.concatenate([y[0].rightids, y[1].rightids, y[2].rightids, y[3].rightids]))
            else:
                name = list(np.concatenate([y[0].rightids, y[1].rightids]))
            self.feaGet_core(i, name, extract, num_thread)

    @ut.timing('feaExtract_10s')
    def feaExtract_10s(self, num_thread, extract: bool = True):
        en = sidekit.IdMap(root + "fea/enroll_idmap_10s.h5")
        te = sidekit.IdMap(root + "fea/test_idmap_10s.h5")
        name = list(np.concatenate([en.rightids, te.rightids]))
        self.feaGet_core(10, name, extract, num_thread)

    def feaGet_core(self, i, name, extract, num_thread=None):
        extra, name_list, channel_list = [], [], []
        if i == 10:
            for j in name:
                name_list.append(j[:-7])  # may is should be -7
                channel_list.append(0 if j[-2] == 'A' else 1)
                extra.append(j[-3:])
        else:
            for j in name:
                name_list.append(j[:-6])
                channel_list.append(0 if j[-1] == 'A' else 1)
                extra.append(j[-2:])
        param3, param4 = '', ''
        if i in [4, 5, 6, 8]:
            param3, param4 = "sre0" + str(i) + "/data/{}.sph", "/fea/fea/sre0" + str(i) + "/{}{}.h5"
        elif i == 10:
            param3, param4 = "sre" + str(i) + "/data/{}.sph", "/fea/fea/sre" + str(i) + "/{}{}.h5"
        elif i in ['cellular', 'phase']:
            param3, param4 = "swb/" + str(i) + "/{}.SPH", "/fea/fea/" + str(i) + "/{}{}.h5"
        self.audio_filename_structure = root + param3
        self.feature_filename_structure = root + param4
        self.extra = extra
        if extract:
            self.save_list(show_list=name_list,
                           channel_list=channel_list,
                           num_thread=num_thread, extra=self.extra)
        fi = os.listdir(root + param4[:-7])
        if i in [4, 5, 6]:
            s = pd.DataFrame(np.array([i[:4] + i[-2:] + ".h5" for i in name]))
        elif i == 8:
            s = pd.DataFrame(np.array([i[:5] + i[-2:] + ".h5" for i in name]))
        elif isinstance(i, str):
            s = pd.DataFrame(np.array([i[:8] + i[-2:] + ".h5" for i in name]))
        elif i == 10:
            fi = ut.listDirRecursive(root + "fea/fea/sre10/")
            fi = [re.sub(".*sre10/", "", i) for i in fi]
            s = pd.DataFrame(np.array([i[:-7] + i[-3:] + ".h5" for i in name]))
        s = s.loc[~s[0].isin(fi)]
        if s.shape[0]:
            print("year:" + str(i) + "complete!!!\nsome file generate failed,need to generate again!!!")
            pk.dump(s, open('./fea_' + str(i), 'wb'))
        else:
            print("year:" + str(i) + "complete!!!\n")

    def feaExtractAgain(self, year):
        rest = pk.load(open('./fea_' + str(year), 'rb'))
        if year in [4, 5, 6]:
            name = [i[:4] + ".sph" + i[4:6] for i in list(rest[0])]
        if year == 8:
            name = [i[:5] + ".sph" + i[5:7] for i in list(rest[0])]
        elif year in ['cellular', 'phase']:
            name = [i[:8] + ".SPH" + i[8:10] for i in list(rest[0])]
        elif year == 10:
            name = [i[:-6] + ".sph" + i[-6:-3] for i in list(rest[0])]
        self.feaGet_core(year, name, True)

    # overwrite save_list method,cause original has some problems in case that cannot complete 'save_list',fuck!!!
    def save_list(self, show_list, channel_list, audio_file_list=None, feature_file_list=None, num_thread=None,
                  extra=None):

        # get the length of the longest list
        max_length = max([len(l) for l in [show_list, channel_list, audio_file_list, feature_file_list]
                          if l is not None])

        if show_list is None:
            show_list = np.empty(int(max_length), dtype='|O')
        if audio_file_list is None:
            audio_file_list = np.empty(int(max_length), dtype='|O')
        if feature_file_list is None:
            feature_file_list = np.empty(int(max_length), dtype='|O')
        # ma=Manager()
        # lock=ma.Lock()
        #
        pool = Pool(num_thread  # num_thread
            # ,initializer=Fea.globalized,initargs=(lock)
        )  # default number of processes is os.cpu_count()

        pool.map(self.extract, list(zip(show_list, channel_list, audio_file_list, feature_file_list, extra)))
        pool.close()
        pool.join()

    def extract(self, *file):
        show, channel, input_audio_filename, output_feature_filename, extra = file[0]
        backing_store = True
        if input_audio_filename is not None:
            self.audio_filename_structure = input_audio_filename
        audio_filename = self.audio_filename_structure.format(show)

        # If the output file name does not include the ID of the show,
        # (i.e., if the feature_filename_structure does not include {})
        # the feature_filename_structure is updated to use the output_feature_filename
        if output_feature_filename is not None:
            self.feature_filename_structure = output_feature_filename

        if extra:
            feature_filename = self.feature_filename_structure.format(show, extra)
        else:
            feature_filename = self.feature_filename_structure.format(show)
        # if os.path.exists(feature_filename):
        #     return
        # Open audio file, get the signal and possibly the sampling frequency
        signal, sample_rate = read_audio(audio_filename, self.sampling_frequency)
        if signal.ndim == 1:
            signal = signal[:, np.newaxis]

        # Process the target channel to return Filter-Banks, Cepstral coefficients and BNF if required
        length, chan = signal.shape

        # If the size of the signal is not enough for one frame, return zero features
        PARAM_TYPE = np.float32
        if length < self.window_sample:
            cep = np.empty((0, self.ceps_number), dtype=PARAM_TYPE)
            energy = np.empty((0, 1), dtype=PARAM_TYPE)
            fb = np.empty((0, self.filter_bank_size), dtype=PARAM_TYPE)
            label = np.empty((0, 1), dtype='int8')

        else:
            # Random noise is added to the input signal to avoid zero frames.
            np.random.seed(0)
            signal[:, channel] += 0.0001 * np.random.randn(signal.shape[0])

            dec = self.shift_sample * 250 * 25000 + self.window_sample
            dec2 = self.window_sample - self.shift_sample
            start = 0
            end = min(dec, length)

            # Process the signal by batch to avoid problems for very long signals
            while start < (length - dec2):

                if self.feature_type == 'mfcc':
                    # Extract cepstral coefficients, energy and filter banks
                    cep, energy, _, fb = mfcc(signal[start:end, channel],
                                              fs=self.sampling_frequency,
                                              lowfreq=self.lower_frequency,
                                              maxfreq=self.higher_frequency,
                                              nlinfilt=self.filter_bank_size if self.filter_bank == "lin" else 0,
                                              nlogfilt=self.filter_bank_size if self.filter_bank == "log" else 0,
                                              nwin=self.window_size,
                                              shift=self.shift,
                                              nceps=self.ceps_number,
                                              get_spec=False,
                                              get_mspec=True,
                                              prefac=self.pre_emphasis)
                elif self.feature_type == 'plp':
                    cep, energy, _, fb = plp(signal[start:end, channel],
                                             nwin=self.window_size,
                                             fs=self.sampling_frequency,
                                             plp_order=self.ceps_number,
                                             shift=self.shift,
                                             get_spec=False,
                                             get_mspec=True,
                                             prefac=self.pre_emphasis,
                                             rasta=self.rasta_plp)

                # Perform feature selection
                label, threshold = self._vad(cep, energy, fb, signal[start:end, channel])
                # print(len(label[label]))
                if len(label) < len(energy):
                    label = np.hstack((label, np.zeros(len(energy) - len(label), dtype='bool')))

                start = end - dec2
                end = min(end + dec, length)

        # Create the HDF5 file
        # Create the directory if it dosn't exist
        dir_name = os.path.dirname(feature_filename)  # get the path
        if not os.path.exists(dir_name) and (dir_name is not ''):
            os.makedirs(dir_name)

        h5f = h5py.File(feature_filename, 'a', backing_store=backing_store, driver='core')
        if "cep" not in self.save_param:
            cep = None
            # cep_mean = None
            # cep_std = None
        if "energy" not in self.save_param:
            energy = None
            # energy_mean = None
            # energy_std = None
        if "fb" not in self.save_param:
            fb = None

        if "vad" not in self.save_param:
            label = None


        cep, fb, label = self.postProc(cep, energy, fb, label)

        write_hdf5(show, h5f,
                   cep, None, None,
                   None, None, None,
                   fb, None, None,
                   None, None, None,
                   label)

        h5f.close()
        pass

    def postProc(self, cep, energy, fb, label):
        cep, _ = self.feaServer.post_processing(np.concatenate([energy[:, np.newaxis], cep], axis=1), label)
        fb, label = self.feaServer.post_processing(fb, label)
        return cep, fb, label

    def _vad(self, cep, log_energy, fb, x, label_file_name=None):

        label, threshold = vad_energy(log_energy, distrib_nb=3,
                                      nb_train_it=8, flooring=0.0001,
                                      ceiling=1.5, alpha=2.5)  # defult is alpha=0.1
        return label, threshold

def main():
    # fea.feaExtractAgain(10)

    # remote
    # num_thread = 48
    # fea = Fea()
    # fea.feaExtract(num_thread,[4,5,6,8],True)

    # local
    num_thread = 32
    fea = Fea()
    fea.feaExtract_10s(num_thread, True)
    fea.feaExtract(num_thread, ['cellular', 'phase'], True)


if __name__ == '__main__':
    main()
