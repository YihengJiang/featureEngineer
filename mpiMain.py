#!/usr/bin/env python
#  -*- coding:utf-8 -*-
import os
import globalVar as glb
from ivMpi_PipeLine import *
import time
import jyh.Utils as ut

@ut.timing("Total")
def main():
    # time.sleep(int(14.5*3600))
    # 1 and 2 cannot sum in one step, they should be splitted to run
    os.system("mpirun --hostfile " + glb.get_codeDir() + "hostList.txt python ivMpi_PipeLine/ubm1.py")
    # ###############################################################################################
    # stat2.main()
    # tv3.main() # tv3 cannot use mpi to speed up computing
    # os.system("mpirun --hostfile " + glb.get_codeDir() + "hostList.txt python ivMpi_PipeLine/iv4.py")
    # result5.main()

if __name__ == '__main__':
    main()
