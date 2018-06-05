#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import IVector
import os

os.environ['SIDEKIT'] = 'theano=false,theano_config=cpu,libsvm=false,mpi=true'


def main():
    print("Begin TV" + ("-" * 60))
    iv = IVector.IV(48)
    # iv.learnTV_mpi() #will conduct error when compute,so i will not use it
    iv.learnTV()


if __name__ == '__main__':
    main()
