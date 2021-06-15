import torch as tc
import torchvision as tv
import numpy as np
import os
import copy
from termcolor import cprint
import torchvision.transforms as transforms
from inspect import stack
import cv2
import time
import datetime as dtm
from datetime import datetime
import random


def choose_device(n=0):
    if tc.cuda.is_available():
        if n is None:
            return tc.device("cuda:0")
        else:
            return tc.device("cuda:"+str(n)[5:])
    else:
        return tc.device("cpu")


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    path_flag = os.path.exists(path)
    if not path_flag:
        os.makedirs(path)
    return path_flag


def save(path, file, data, names):
    mkdir(path)
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    tc.save(tmp, os.path.join(path, file))


def load(path_file, names=None, device='cpu'):
    if os.path.isfile(path_file):
        if names is None:
            data = tc.load(path_file)
            return data
        else:
            tmp = tc.load(path_file, map_location=device)
            if type(names) is str:
                data = tmp[names]
                return data
            elif (type(names) is list) or (type(names) is tuple):
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                return tuple(data)
    else:
        return False


def project_path(project='T-Nalg/'):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    return cur_path[:cur_path.find(project) + len(project)]


def output_txt(x, filename='data'):
    np.savetxt(filename + '.txt', x)


def print_dict(a, keys=None, welcome='', style_sep=': ', end='\n', file=None):
    express = welcome
    if keys is None:
        for n in a:
            express += n + style_sep + str(a[n]) + end
    else:
        if type(keys) is str:
            express += keys.capitalize() + style_sep + str(a[keys])
        else:
            for n in keys:
                express += n.capitalize() + style_sep + str(a[n])
                if n is not keys[-1]:
                    express += end
    if file is None:
        print(express)
    else:
        fprint(express, file)
    return express


def combine_dicts(dic1, dic2):
    # dic1中的重复key值将被dic2覆盖
    return dict(dic1, **dic2)


def random_dates(num, start_date, end_date, if_weekday=True, dates=None, exclude_dates=None):
    # start_date = (year, month, day)
    start = time.mktime(start_date + (0, 0, 0, 0, 0, 0))
    end = time.mktime(end_date + (0, 0, 0, 0, 0, 0))
    if dates is None:
        dates = list()
    if exclude_dates is None:
        exclude_dates = list()
    it_max = 0
    while (len(dates) < num) and (it_max < num * 40):
        t = random.randint(start, end)
        date_touple = time.localtime(t)
        date_touple = time.strftime("%Y%m%d", date_touple)
        # print(date_touple)
        # print(datetime.strptime(date_touple, "%Y%m%d").weekday())
        if (datetime.strptime(date_touple, "%Y%m%d").weekday() < 5) or (
                not if_weekday):
            if (date_touple not in dates) and (date_touple not in exclude_dates):
                dates.insert(0, date_touple)
            # date = list(set(date))
        it_max += 1
    return dates


def fprint(content, file=None, print_screen=True, append=True):
    if file is None:
        file = './record.log'
    if append:
        way = 'ab'
    else:
        way = 'wb'
    with open(file, way, buffering=0) as log:
        log.write((content + '\n').encode(encoding='utf-8'))
    if print_screen:
        print(content)


def warning(string, if_trace_stack=False):
    cprint(string, 'magenta')
    if if_trace_stack:
        trace_stack(3)


def trace_stack(level0=2):
    # print the line and file name where this function is used
    info = stack()
    ns = info.__len__()
    for ns in range(level0, ns):
        cprint('in ' + str(info[ns][1]) + ' at line ' + str(info[ns][2]), 'green')


def now(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d %H:%M:%S'
    return time.strftime(fmt, time.localtime(time.time()))


def kron(mat1, mat2):
    return tc.einsum('ab,cd->acbd', mat1, mat2).reshape(
        mat1.shape[0] * mat2.shape[0], mat1.shape[1] * mat2.shape[1])





