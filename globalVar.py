#!/usr/bin/env python
# -*- coding:utf-8 -*-
class global_var():
    '''需要定义全局变量的放在这里，最好定义一个初始值'''
    root = '/home/jyh/D/jyh/data/'
    # root = '/home/jyh/data/'


# 对于每个全局变量，都需要定义get_value和set_value接口
def set_root(root):
    global_var.name = root


def get_root():
    return global_var.root
