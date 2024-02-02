# Artificial_Neural_Network.py
# Trains ANN on Fisher's Iris data then classifies user input plants
# A6 Gardens Of Heaven assignment.
#
# CS 131 - Artificial Intelligence
#
# Written by - Vivian Lau vlau02
# Last modified - 12/20/2023

import math
import copy
import random # for shuffling and generating random weights
import numpy as np # for calculating exponential e^x function values

TRAIN_RATIO = 0.6
VALIDATE_RATIO = 0.2
TEST_RATIO = 0.2

class ArtificialNeuralNetwork:
    
    # Takes in training data, learning rate, and number of runs to train
    # ANN model
    def __init__(self, trainDataIn, learnRateIn, numRunsIn):
        # Data is expected to have following attributes (in cm) in order of:
        # Sepal length, Sepal width, Petal length, Petal width, Class
        self.trainData = [trainDataIn, []]
        self.validateData = [[], []]
        self.testData = [[], []]
        self.prepData() # puts given data into each set
        
        self.layers = [4, 8, 10, 3]
        # starting with 4 attributes (for each node
        # choose some number of hidden layers
        # end with 3 values that representing encoding of class (outcome)
        
        self.weights = None
        self.initWeights() # random weights are made based on # of layers
        
        self.learnRate = learnRateIn # given by user
        self.numRuns = numRunsIn # given by user
        
        self.train() # trains neural network based on data given above
    
    # Formats the data to make easier to train model on
    # Includes modifying attribute data to floats, separating and encoding
    # the desired outcome (class) into another column, and allocating
    # specified ratios of training, validation, and testing data ratios
    # (global variables)
    def prepData(self):
        # shuffle data
        random.shuffle(self.trainData[0])
        
        totalNumData = len(self.trainData[0])
        dataSet = self.trainData
        
        validateStart = math.floor(TRAIN_RATIO*totalNumData)
        validateEnd = validateStart + math.floor(VALIDATE_RATIO*totalNumData)
        # seperate each classification from the other attributes
        for i in range(totalNumData):
            if i == validateStart:
                dataSet = self.validateData
            elif i == validateEnd:
                dataSet = self.testData
            
            classification = self.trainData[0][i].pop() # get classification
            # format rest of attributes to floats
            self.trainData[0][i] = [float(x) for x in self.trainData[0][i]]
            
            if classification == 'Iris-setosa': # encode desired classification
                dataSet[1].append([1,0,0])
            elif classification == 'Iris-versicolor':
                dataSet[1].append([0,1,0])
            elif classification == 'Iris-virginica':
                dataSet[1].append([0,0,1])
        self.validateData[0] = self.trainData[0][validateStart:validateEnd]
        self.testData[0] = self.trainData[0][validateEnd:]
        self.trainData[0] = self.trainData[0][:validateStart]
        
    # Initial weightings are random generated (both positive and negative weights)
    def initWeights(self):
        self.weights = list()
        for i in range(1, len(self.layers)):
            # randomly choose weights
            weight = [[random.uniform(-1.0, 1.0) \
                       for j in range(self.layers[i-1]+1)] \
                       for k in range(self.layers[i])]
            self.weights.append(weight)
    
    # Activation function chosen to be logistic
    def calcActivation(self, p):
        return [(1/(1+np.exp(-x))) for x in p]
        #return max(0, p) # RELU
    
    # Forward Propagation
    # Done for a single node and determines the expected class
    # for the given attribute data (x) and specified weights (self.weights)
    # when called
    def forwardProp(self, x):
        outputs = [x]
        v = x
        for i in range(len(self.weights)):
            newV = []
            for j in range(len(self.weights[i])):
                temp = 0
                for k in range(len(v)):
                    temp += v[k] * self.weights[i][j][k]
                newV.append(temp)
            activation = self.calcActivation(newV)
            outputs.append(activation)
            v = [1,] + activation
        return outputs

    # Backward propagation
    # desired outcomes for each of the predicted outputs are parameters
    # used to caclulate the error which is then propagated backward
    # for each layer by calculating the local gradient of the neurons
    def backwardProp(self, desire, outputs):
        curError = [desire[i]-outputs[-1][i] for i in range(len(desire))]
        for i in range(len(outputs)-1, 0, -1):
            curActive = outputs[i] # current layer we are looking at
            prev = [1,] + outputs[i-1] # previous layer
            if i == 1: # no more previous layers after
                prev = outputs[0]
               
            # get derivative (change in output)
            deriv = [x*(1-x) for x in curActive]
            deriv = [curError[x]*deriv[x] for x in range(len(curError))]
            for j in range(len(self.weights[i-1])):
                for k in range(len(self.weights[i-1][0])):
                    temp = self.learnRate*deriv[j]*prev[k]
                    self.weights[i-1][j][k] += temp
            
            curError = [] # calculate error for next preceding layer
            for j in range(1,len(self.weights[i-1][0])):
                temp = 0
                for k in range(len(deriv)):
                    temp += deriv[k] * self.weights[i-1][k][j]
                curError.append(temp)
    
    # Predicts the classification of a plant with the given attributes (x)
    # expects data in order of Sepal length, Sepal width, Petal length, 
    # and Petal width all in the units of cm.
    def predict(self, x):
        x = [1,]+ x
        outputs = self.forwardProp(x)
        results = outputs[-1] # become 1d array
        maxIndex = 0
        for i in range(1, len(results)):
            if(results[i] > results[maxIndex]):
                maxIndex = i
        y = [0 for i in range(len(results))]
        y[maxIndex] = 1
        return y

    # Calculated accuracy of a given dataset (training, validation,
    # or testing) by counting if the predicted is correct
    def getAccuracy(self, dataSet):
        numRight = 0
        for i in range(len(dataSet[0])):
            if dataSet[1][i] == self.predict(dataSet[0][i]):
                numRight += 1 # count number of right predictions
        return round(numRight/len(dataSet[0]), 2)
    
    # Trains the model by calling forward propagation and backward
    # propagation until the best model is found or reaches the
    # max number of runs (as specified by user when initializing
    # instance of the model)
    # Also takes care of printing the progress of the model while training
    def train(self):
        print("-----------------------------------------------")
        prevValidError = 1
        bestWeights = copy.deepcopy(self.weights)
        hitMin = False
        for run in range(1, self.numRuns+1):
            for i in range(len(self.trainData[0])):
                x = [1,] + self.trainData[0][i]
                outputs = self.forwardProp(x)
                self.backwardProp(self.trainData[1][i], outputs)
                
            curValidAccuracy = self.getAccuracy(self.validateData)
            # stop early before overtrained but not too early
            if (1-curValidAccuracy) > prevValidError and curValidAccuracy > 0.9:
                hitMin = True
            elif hitMin:
                print("Already found best model,",\
                      "stopping early to prevent overtraining")
                self.weights = bestWeights # most recent with highest accuracy
                print("Layer: "+str(run))
                print("Training Accuracy:", self.getAccuracy(self.trainData))
                print("Validating Accuracy:", curValidAccuracy)
                print("-----------------------------------------------")
                break
            else:
                bestWeights = copy.deepcopy(self.weights)
                prevWeight = 4
                prevValidError = prevValidError * prevWeight + (1-curValidAccuracy)
                prevValidError /= (prevWeight+1) # normalize, average out
            
            if(run % 25 == 0): # print every 25 runs
                print("Layer: "+str(run))
                print("Training Accuracy:", self.getAccuracy(self.trainData))
                print("Validating Accuracy:", curValidAccuracy)
                print("-----------------------------------------------")
            
        print("Testing Accuracy: ", self.getAccuracy(self.testData))
        print("-----------------------------------------------")
        
        