# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:04:34 2019

@author: kzt9qh
"""

import os
import timeit
import shutil

start = timeit.default_timer()

address_Images = r"V:\2019\WIP\AV_GamesOfDL_Image_Classification\IN\images"
address_TrainTest = r'V:\2019\WIP\AV_GamesOfDL_Image_Classification'

trainCSVFilePath = os.path.join(address_TrainTest, "train.csv")
testCSVFilePath = os.path.join(address_TrainTest, "test.csv")

Train_files = []
if os.path.exists(trainCSVFilePath):
    with open(trainCSVFilePath) as csvFile:
        Train_files = csvFile.readlines()
Train_files = [item.split(",") for item in Train_files[1:]]
        
Test_files = []
if os.path.exists(testCSVFilePath):
    with open(testCSVFilePath) as csvFile:
        Test_files = csvFile.readlines()
Test_files = [item.split(",") for item in Test_files[1:]]
        
for picNumber in range(0,len(Train_files)):
    picName = Train_files[picNumber][0]
    shutil.copy(address_Images+'/'+picName,address_TrainTest+'/Images_Train/'+picName)

for picNumber in range(0,len(Test_files)):
    picName = Test_files[picNumber][0][:-1]
    shutil.copy(address_Images+'/'+picName,address_TrainTest+'/Images_Test/'+picName)

stop = timeit.default_timer()
print('RUN Time: ', stop - start)