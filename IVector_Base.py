#!/usr/bin/env python
# -*- coding:utf-8 -*-
from abc import ABCMeta, abstractmethod


class IVector_Base(metaclass=ABCMeta):
    '''There are 5 steps involved:

     1. training a UBM from background data
     2. learning a total variability subspace from background statistics
     3. training a Gaussian PLDA model with development i-vectors
     4. scoring verification trials with model and test i-vectors
     5. computing the performance measures (e.g., EER and confusion matrix)
    '''

    '''Step1: Create the universal background model from all the training speaker data'''

    @abstractmethod
    def trainUBM(self, data=None):
        pass

    '''Step2.1: Calculate the statistics from train datat set needed for the iVector model.'''

    @abstractmethod
    def calStat(self, data=None):
        pass

    '''Step2.2: Learn the total variability subspace from all the train speaker data.'''

    @abstractmethod
    def learnTV(self, data=None):
        pass

    '''Step2.3:Now compute the development ivectors of train data set for each speaker and channel.  The result is size tvDim x nSpeakers x nChannels.'''

    @abstractmethod
    def computeDevIV(self, data=None):
        pass

    '''Step3.1:do LDA on the development iVectors to find the dimensions that matter.'''

    @abstractmethod
    def trainLDA(self, data=None):
        pass

    '''Step3.2: Now train a Gaussian PLDA model with development i-vectors'''

    @abstractmethod
    def trainGPLDA(self, data=None):
        pass

    '''Step4: now we have the channel and LDA models. Let's compute ivector and do lda with enrollment and test,respectively'''

    @abstractmethod
    def computeIVAndDoLDA_OnEnrollOrTest(self, data=None):
        pass

    '''Step5: Now score the models with all the test data.'''

    @abstractmethod
    def scoreAndVisualization(self, data=None):
        pass
