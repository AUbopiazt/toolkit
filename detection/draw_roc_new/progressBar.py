#coding=utf-8
from tqdm import tqdm
from tqdm._tqdm import trange

class progressBar(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, work_name):
        self.work_name = work_name

    def getName(self):
        return self.work_name

    def setName(self, work_name):
        self.work_name = work_name

    def trange(self, num):
        return trange(num)

    def tlist(self, list_):
        tl = tqdm(list_)
        tl.set_description(self.work_name+']')
        return tl







