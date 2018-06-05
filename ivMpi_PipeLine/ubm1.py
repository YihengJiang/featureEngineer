#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import IVector
import os

os.environ['SIDEKIT'] = 'theano=false,theano_config=cpu,libsvm=false,mpi=true'


def main():
    print("Begin UBM" + ("-" * 60))
    iv = IVector.IV(48)
    iv.trainUBM_mpi()
    # iv.trainUBM()

if __name__ == '__main__':
    main()
