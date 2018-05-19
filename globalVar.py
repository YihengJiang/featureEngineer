#!/usr/bin/env python
# -*- coding:utf-8 -*-
class global_var():
    '''需要定义全局变量的放在这里，最好定义一个初始值'''
    _root = '/home/jyh/D/jyh/data/'
    # root = '/home/jyh/data/'
    _codeDir = '/home/jyh/pycharmProjects/feartureEngineer/'

    _count = 0
# 对于每个全局变量，都需要定义get_value和set_value接口
def get_root():
    return global_var._root


def get_codeDir():
    return global_var._codeDir


def set_count():
    global_var._count += 1
    if global_var._count % 50 == 1:
        print(global_var._count)


def get_count():
    return global_var._count


def reset_count():
    global_var._count = 0
