import csv
import math
from operator import itemgetter
import random
import sys
import argparse

'''
with open("knn_train.csv","r") as f:
    reader = csv.reader(f)
  
    features = next(reader)
    # print(features)
  
    DataList = []
    for row in reader:
        DataList += [row]
        print(row)
'''

def manhattanDistance(list1, list2):

    if len(list1) != len(list2):
        raise Exception('Length of Lists are not equal')
    
    distance = 0
    for i in range(len(list1)-1):
        distance += abs(int(list1[i+1]) - int(list2[i+1]))
        
    return distance

def euclideanDistance(list1, list2):

    if len(list1) != len(list2):
        raise Exception('Length of Lists are not equal')

    distance = 0
    for i in range(len(list1)-1):
        distance += pow((int(list1[i+1]) - int(list2[i+1])), 2)

    return math.sqrt(distance)

def maxNormDistance(list1, list2):
    
    if len(list1) != len(list2):
        raise Exception('Length of Lists are not equal')

    distance = []
    for i in range(len(list1)-1):
        distance.append(abs(int(list1[i+1]) - int(list2[i+1])))

    return max(distance)

#testData1 = DataList[1].copy()
#testData2 = DataList[2].copy()
#testData1 .remove('4000')

#print(testData1)
#print(testData2)

#x = euclideanDistance(testData1, testData2)
#print(x)

def kNearestNeighbors(trainingSet, testInstance, k, method):
    '''
    with open("knn_train.csv","r") as f:
        reader = csv.reader(f)
        features = next(reader)
        # print(features)
  
        DataList = []
    '''

    distanceList = []
        
    #Finds distances of testValue to TrainingSet
    if method == "L1":

        for i in trainingSet:
            #print(i)
            dist = manhattanDistance(testInstance, i)
            #print(dist)
            distanceList.append((i[0], dist))
    
    elif method == "Linf":
        for i in trainingSet:
            #print(i)
            dist = maxNormDistance(testInstance, i)
            #print(dist)
            distanceList.append((i[0], dist))
    
    else:
        for i in trainingSet:
            #print(i)
            dist = euclideanDistance(testInstance, i)
            #print(dist)
            distanceList.append((i[0], dist))
    
    #print(distanceList)

    #Sort for nearest distances to get neighbors
    x = sorted(distanceList, key=itemgetter(1))
        
    neighbors = []
        
    for i in range(k):
        neighbors.append(x[i])

    return neighbors

def prediction(neighbors):
    counter = 0

    for i in neighbors:
        counter += int(i[0])
   
    #print(counter)
    if counter == 0:
        return random.choice((-1, 1))
    elif counter > 0:
        return 1
    else:
        return -1
    
def main():

    with open("knn_train.csv","r") as f:
        reader = csv.reader(f)
        features = next(reader)
        # print(features)
  
        DataList = []

        for row in reader:
            DataList += [row]

    with open("knn_test.csv", "r") as g:
        reader2 = csv.reader(g)
        features2 = next(reader2)

        DataList2 = []

        for row in reader2:
            DataList2 += [row]

        #print(DataList)
        #print(DataList2)

        #testData1 = DataList[1].copy()
        #testData2 = DataList[2].copy()
        #testData1 .remove('4000')

        #print(testData1)
        #print(testData2)

        #euclidianTest = euclideanDistance(testData1, testData2)
        #print(euclidianTest)
        
        #manhattanTest = manhattanDistance(testData1, testData2)
        #print(manhattanTest)

        #maxNormTest = maxNormDistance(testData1, testData2)
        #print(maxNormTest)

        testInstance = []
        testInstance = DataList2[0].copy()
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--K")
        parser.add_argument("--method")
        args = parser.parse_args()
        #print(args.K)

        print("The List of Neighbors are:")
        x = kNearestNeighbors(DataList, testInstance, int(args.K), str(args.method))
        print(x)
        print()

        result = prediction(x)
        print("The predicted label will be: %d" % result)
        print()


        #Answers Problem 4 No.6
        """        
        for i in range(10):

            testInstance = DataList2[i].copy()
            print("THIS IS FOR INSTANCE %d" % (i+1))
            #manhattanTest = maxNormDistance(DataList[0], DataList2[0])
            #print(manhattanTest)

            x = kNearestNeighbors(DataList, testInstance, 1, "L2")
            #print(x)

            result = prediction(x)
            print(result)
 
            x = kNearestNeighbors(DataList, testInstance, 3, "L2")
            #print(x)

            result = prediction(x)
            print(result)
 
            x = kNearestNeighbors(DataList, testInstance, 5, "L2")
            #print(x)

            result = prediction(x)
            print(result)
 
            x = kNearestNeighbors(DataList, testInstance, 720, "L2")
            #print(x)

            result = prediction(x)
            print(result)
            print()
        """    
        
        #Answers Problem 4 No.7
        """
        for i in range(10):

            testInstance = DataList2[i].copy()
            print("THIS IS FOR INSTANCE %d" % (i+1))
            #manhattanTest = maxNormDistance(DataList[0], DataList2[0])
            #print(manhattanTest)
            
            x = kNearestNeighbors(DataList, testInstance, 1, "L1")
            #print(x)

            result = prediction(x)
            print("This result using L1 is %s" % result)
 
            x = kNearestNeighbors(DataList, testInstance, 3, "L2")
            #print(x)

            result = prediction(x)
            print("The result using L2 is %s" % result)
 
            x = kNearestNeighbors(DataList, testInstance, 5, "Linf")
            #print(x)

            result = prediction(x)
            print("The result using Linf is %s" % result)
            print() 
        """    
if __name__ == "__main__":
    main()
