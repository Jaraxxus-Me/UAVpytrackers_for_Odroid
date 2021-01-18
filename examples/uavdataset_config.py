# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 09:18:12 2020

@author: Jaraxxus
"""
import os
class UAVDatasetConfig:
    def __init__(self,vediopath):
        self.path=vediopath
        frames={}
        names=os.listdir(vediopath)
        for name in names:
            frames[name]=[1,len(os.listdir(os.join(self.path,name)))]