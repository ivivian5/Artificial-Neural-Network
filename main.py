#
# main function for A6 Gardens Of Heaven assignment.
# Calls artificial neural network classifier to classify user input plants based on 
# training on Fisher's Iris data.
#
# CS 131 - Artificial Intelligence
#
# Written by - Vivian Lau vlau02
# Last modified - 12/20/2023

from Artificial_Neural_Network import ArtificialNeuralNetwork

# begin format
print ("----------------------------------------------------------------------")
print ("Welcome to Vivian's trained Artificial Neural Network for \n" + \
       "classifying iris plants their sepal length, sepal width, petal \n" +\
       "length, and petal width!")
print ("----------------------------------------------------------------------")

irisDataFile = None
irisData = list()
fileFound = True

try:
    print("\nGetting Iris Data...")
    irisDataFile = open('irisData.txt', 'r')
    print("Success: Iris Data obtained for training.\n")
    
    irisDataFile = irisDataFile.readlines()
    for line in irisDataFile: # each iris plant
        line = line.strip() # remove end carriage
        line = [x for x in line.split(',')] # seperating attributes
        irisData.append(line) # add plant
    irisData.pop() # remove the endFile
except FileNotFoundError:
    fileFound = False # Don't train ANN
    print("Error: File for training iris data not found, cannot execute.")
    print("File must be named \"irisData.txt\" and in same directory.")

if fileFound:
    print("\nTraining Artificial Neural Network...")
    ann = ArtificialNeuralNetwork(irisData, 0.1, 250)
    print("Success: Artificial Neural Network is trained.\n")
    
    done = False
    while not done:
        print("-----------------------------------------------")
        userInput = input('Enter attribute values (in cm) for iris plant \n'+\
                          'in order of: (1)Sepal length, (2)Sepal width, \n'+\
                          '(3)Petal length, and (4)Petal width seperated by \n'+\
                          'commas (or \"done\" if no more): ')
        if (userInput == 'done'):
            done = True
        else:
            userInput = userInput.strip() # remove end carriage
            userInput = [x for x in userInput.split(',')] # seperating attributes
            try:
                if len(userInput) == 4:
                    userInput = [float(x) for x in userInput]
                    predictClass = ann.predict(userInput)
                    if predictClass == [1, 0, 0]:
                        print("\nI think this flower is a Iris-setosa")
                    elif predictClass == [0, 1, 0]:
                        print("\nI think this flower is a Iris-versicolor")
                    else:
                        print("\nI think this flower is a Iris-virginica")
                else:
                    raise ValueError
            except ValueError:
                print("Error: Invalid input, please try again")

# end format
print ("----------------------------------------------------------------------")
print ("Thanks for using Vivian's Artificial Neural Network.")
print ("----------------------------------------------------------------------")
    