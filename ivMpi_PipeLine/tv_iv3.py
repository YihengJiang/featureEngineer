#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import IVector
import os

os.environ['SIDEKIT'] = 'theano=false,theano_config=cpu,libsvm=false,mpi=true'


def main():
    iv = IVector.IV(48)
    iv.learnTV_mpi().extractIV_mpi()


if __name__ == '__main__':
    main()
