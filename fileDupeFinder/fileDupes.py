#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 07:28:16 2018

@author: phil
"""

import os
import pandas as pd
import re

def fileDump(rootDir):
    allFiles = []
    allDirs = []
    
    for e in os.walk(rootDir):
        #   Remove hidden files and folders
        tFiles = [a for a in e[2] if not re.match(r'^\..+',a)]
        allFiles += tFiles
        allDirs += [e[0]] * len(tFiles)
    #   Print dimensions... Originally for debugging purposes, but I left it.
    print(len(allFiles),len(allDirs))
    #   Return in a data frame, because it's easier to csv it.
    fileList = pd.DataFrame({'file': allFiles, 'dir': allDirs})
    fileList = fileList.sort_values(by = 'file')
    return fileList

def printDupes(rootDir):
    fileList = fileDump(rootDir)
    dupes = fileList[fileList.duplicated(subset='file',keep=False)]
    print(dupes.size)
    dupes.to_csv('dupedFiles.csv')


