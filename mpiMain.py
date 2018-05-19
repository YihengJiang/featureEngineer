#!/usr/bin/env python
#  -*- coding:utf-8 -*-
import os
import globalVar as glb
from ivMpi_PipeLine import *
import jyh.Utils as ut


@ut.timing("Total")
def main():
    # os.system(
    #     "mpirun --hostfile " + glb.get_codeDir() + "hostList.txt /home/jyh/anaconda3/bin/python3.6 IVector.py")
    print("===================================================================================")
    os.system(
        "mpirun --hostfile " + glb.get_codeDir() + "hostList.txt /home/jyh/anaconda3/bin/python3.6 ivMpi_PipeLine/ubm1.py")
    print("===================================================================================")
    stat2.main()
    # print("===================================================================================")
    # os.system("mpirun --hostfile " + glb.get_codeDir() + "hostList.txt /home/jyh/anaconda3/bin/python3.6 ivMpi_PipeLine/tv_iv3.py")
    # print("===================================================================================")
    # result4.main()
    # print("===================================================================================")

if __name__ == '__main__':
    main()
