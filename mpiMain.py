#!/usr/bin/env python
#  -*- coding:utf-8 -*-
import os
import globalVar as glb


def main():
    os.system("mpirun --hostfile " + glb.get_codeDir() + "hostList.txt /home/jyh/anaconda3/bin/python3.6 FeaGet.py")


if __name__ == '__main__':
    main()
