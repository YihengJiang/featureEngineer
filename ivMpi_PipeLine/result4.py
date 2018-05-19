#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import IVector
import os

os.environ['SIDEKIT'] = 'theano=false,theano_config=cpu,libsvm=false,mpi=true'


def main():
    iv = IVector.IV(48)
    iv.LDA_WCCN_cos_Score().mahalanobis_distance_Score().two_covariance_Score().PLDA_Score().PLDA_LDA_Score().visualization()


if __name__ == '__main__':
    main()
