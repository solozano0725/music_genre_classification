from near_neighnbors import NearestNeighbors
from python_speech_features import mfcc
import scipy.io.wavfile as wav

import os
import pickle
import random 
import matplotlib as mpl
import numpy as np

class TrainingModel:

    def __init__(self, pathSet, fileName):
        self.dataset = []
        self.trainingSet = []
        self.testSet = []
        self.pathSet = pathSet
        self.fileName = fileName

    #1.Load all record songs into "my.dat"
    def loadRecordSongs(self, folder): 
        obj = os.scandir(folder) 
        for entry in obj : 
            if entry.is_dir(): 
                f = f"{folder}\\{entry.name}"
                self.extractFeaturesToBinaryFile(f)

    #2.Extract features from the dataset and dump these features into a binary .dat file “my.dat”:
    def extractFeaturesToBinaryFile(self, pathSetFolder):
        f= open(self.fileName ,'wb')
        i=0
        for file in os.listdir(pathSetFolder):  
            (rate,sig) = wav.read(pathSetFolder+"/"+file)
            mfcc_feat = mfcc(sig,rate ,winlen=0.020, appendEnergy = False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix , covariance , i)
            pickle.dump(feature , f)
        f.close()
        return f
    
    #3.Train and test split on the dataset:           
    def getTrainTestData(self, split, trSet , teSet):
        for x in range(len(self.dataset)):
            if random.random() < split :      
                trSet.append(self.dataset[x])
            else:
                teSet.append(self.dataset[x])  
    
    #Load dataset from file "my.dat"
    def loadDataset(self, fileName):
        with open(fileName, 'rb') as f:
            while True:
                try:
                    self.dataset.append(pickle.load(f))
                except EOFError:
                    f.close()
                    break 

    def trainingModel(self, split):
        if(os.path.exists(self.fileName)==False or os.path.getsize(self.fileName)==0):
            self.loadRecordSongs(self.pathSet)
            self.training(split)
        else:
            self.training(split)
        return self.trainingSet, self.testSet

    def training(self,split):
        self.loadDataset(self.fileName)
        self.getTrainTestData(split, self.trainingSet, self.testSet)

    def get_dataset(self):
        return self.dataset
    
    def get_trainigset(self):
        return self.trainingSet

    def get_testset(self):
        return self.testSet