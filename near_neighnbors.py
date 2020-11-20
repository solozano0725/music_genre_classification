from tempfile import TemporaryFile
import numpy as np
import os
import pickle
import random 
import operator
import math

class NearestNeighbors:

    def __init__(self, trainingSet, instance, k):
        self.__neighbors = self.getNeighbors(trainingSet, instance, k)
        self.__classes = self.nearestClass(self.__neighbors)

    #Get the distance between feature vectors and find neighbors
    def distance(self, instance1, instance2, k ):
        distance = 0 
        
        mm1 = instance1[0] 
        cm1 = instance1[1]

        mm2 = instance2[0]
        cm2 = instance2[1]

        distance  = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 

        distance += (np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 

        distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))

        distance -= k

        return distance

    
    def getNeighbors(self, trainingSet, instance, k):
        distances = []
        neighbors = []

        for x in range (len(trainingSet)):
            dist = self.distance(trainingSet[x], instance, k ) + self.distance(instance, trainingSet[x], k)
            distances.append((trainingSet[x][2], dist))
        distances.sort(key=operator.itemgetter(1))
        
        for x in range(k):
            neighbors.append(distances[x][0])
        print(distances)
        return neighbors

    #Identify the nearest neighbors
    def nearestClass(self, neighbors):
        classVote = {}
        for x in range(len(neighbors)):
            print(len(neighbors))
            response = neighbors[x]
            print(response)
            if response in classVote:
                classVote[response]+=1 
            else:
                classVote[response]=1

        sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
        print(sorter[0][0])
        return sorter[0][0]
    
    def get_neighbors(self):
        return self.__neighbors 

    def get_classes(self):
        return self.__classes
