# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
import sys

import util
import classificationMethod
import math


# THIS IS THE FILE EMANUEL TESFA WORKED ON

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.finalProb = None
        self.priorProb = None

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in list(datum.keys()) ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def calculateAcc(self, guesses, actual):
        '''
        Calculate the accuracy of guesses against acutal
        :return accuracy
        '''
        count = 0
        for g in range(len(guesses)):
            if(guesses[g] == actual[g]):
                count += 1
        return count/len(actual)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"

        priorProb = {0: 0, 1: 0, 2: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        indexVar = { }
        validationVar = { }
        condProb = { }
        finalProb = { }
       

        # calc prior prob
        for item in trainingLabels:
            if item == 0:
                priorProb[0] += 1
            if item == 1:
                priorProb[1] += 1
            if item == 2:
                priorProb[2] += 1
            if item == 3:
                priorProb[3] += 1
            if item == 4:
                priorProb[4] += 1
            if item == 5:
                priorProb[5] += 1
            if item == 6:
                priorProb[6] += 1
            if item == 7:
                priorProb[7] += 1
            if item == 8:
                priorProb[8] += 1
            if item == 9:
                priorProb[9] += 1

        for i in priorProb:
            priorProb[i] /= len(trainingLabels)
        self.priorProb = priorProb

        # get where label occurs

        for i in range(0, len(trainingLabels)):  # key is label numb and value is list at which index contains label
            if trainingLabels[i] not in indexVar:
                indexVar[trainingLabels[i]] = []
                indexVar[trainingLabels[i]].append(i)
            else:
                indexVar[trainingLabels[i]].append(i)

        self.features.sort()

        for label in self.legalLabels:  # from label 0 - 9
            condProb[label] = util.Counter()
            for index in indexVar[label]:
                for feature in self.features:
                    if trainingData[index][feature] == 1:
                        condProb[label][feature] += 1


        for i in range(0, len(validationLabels)):  # key is label numb and value is list at which index contains label
            if validationLabels[i] not in validationVar:
                validationVar[validationLabels[i]] = [i]

            else:
                validationVar[validationLabels[i]].append(i)

       # print(validationVar)

        # item in kgrid
            # iterate through label
                # iterate through feature and calc prob given K value
        bestK = {}
        for k in kgrid:
            self.k = k
            for label in self.legalLabels:
                finalProb[label] = {}
                for feature in self.features:
                    finalProb[label][feature] = ((condProb[label][feature]+self.k) / (len(indexVar[label])+(self.k*2)))
            self.finalProb = finalProb
            guesses = []
            for datum in validationData:
                posterior = self.calculateLogJointProbabilities(datum)
                guesses.append(posterior.argMax())
            accuracy = self.calculateAcc(guesses,validationLabels)
            if accuracy not in bestK:
                bestK[accuracy] = k
            # print('k',self.k)
            # print('accuracy',accuracy)

        self.k = bestK[max(bestK.keys())]
        for label in self.legalLabels:
            finalProb[label] = util.Counter()
            for feature in self.features:
                finalProb[label][feature] = ((condProb[label][feature]+self.k) / (len(indexVar[label])+(self.k*2)))

        self.finalProb = finalProb

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        self.features.sort()
        for label in self.legalLabels:
            sum = 0
            for feature in self.features:
                # print(self.finalProb[label][feature])
                if datum[feature] == 0:
                    # if(self.finalProb[label][feature] == 0):
                    #     print('(1 - self.finalProb[label][feature])',((1 - self.finalProb[label][feature])))
                    sum += math.log((1 - self.finalProb[label][feature]))
                else:
                    sum += math.log(self.finalProb[label][feature])

            logJoint[label] = math.log(self.priorProb[label]) + sum

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        odds = util.Counter()
        for feature in self.features:
            odds[feature] = self.finalProb[label1][feature] / self.finalProb[label2][feature]
        bestFeatures = odds.sortedKeys()
        for i in range(100):
            featuresOdds.append(bestFeatures[i])

        return featuresOdds
