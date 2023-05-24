__authors__ = ['1636290, 1631153, 1636589']
__group__ = ['DJ.10']

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        train_data = train_data.astype(float) #convertimos a float por si acaso
        
        # Reshape  PxMxN -> Px(D=M*N)
        if (len(train_data.shape) == 3):
            P, M, N = train_data.shape
            D = M*N
        else:
            P, M, N, num = train_data.shape
            D = M*N*num
        
        train_data = train_data.reshape((P, D))
        
        # Assign train_data to self.train_data
        self.train_data = train_data
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

    def get_k_neighbours(self, test_data, k):
        
        if (len(test_data.shape) == 3):
            P, M, N = test_data.shape
            D = M*N
        else:
            P, M, N, num = test_data.shape
            D = M*N*num
        
        test_data = test_data.reshape(P,D)
        
        dist = cdist(test_data, self.train_data, 'euclidean')
        #dist = cdist(test_data, self.train_data, 'minkowski')
        #dist = cdist(test_data, self.train_data, 'sqeuclidean')
        #ni idea de cual es el mejor, son las unicas que funcionan por ahora.
        
        proxima = dist.argsort(axis = 1)[:, :k]
        
        label = []
        for i in proxima:
            label.append(self.labels[i]) 
        
        self.neighbors = np.array(label)

        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #creates the two arrays
        resultat = []
        percents = []
        for i in self.neighbors:
            #looks which one is the most voted and its percentage
            #fills the array of class most voted
            votacions = np.array([np.count_nonzero(i == c) for c in i])
            major = np.argmax(votacions)
            resultat.append(i[major])
            #fills the array of percentages
            percmajor = votacions[major] / len(i)
            percents.append(percmajor)
        #turns list resultat into an array
        return np.array(resultat)
        
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
