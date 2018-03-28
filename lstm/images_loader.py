#!/usr/bin/python3.6

from os import listdir
from os.path import isfile, join
from os import walk
from os import mkdir
import numpy as np
import json
import sys
import cv2

def keySort(string):
    return string[-9:-4]

def one_hot_class(boolClass):
    if boolClass:
        return [1,0]
    else:
        return [0,1]

class ImageLoader:

    def __init__(self,inputImagesFolder, inputLabelsFile, newSize, batchSize, sequenceSize):

        self.inputImagesFolder = inputImagesFolder
        self.inputLabelsFile = inputLabelsFile
        self.newSize = newSize
        self.batchSize = batchSize
        self.sequenceSize = sequenceSize

        self.inputsList = []
        self.framesNumber = 0
        self.lastRange = 0

        with open(inputLabelsFile) as f:
            data = json.loads(f.read())

        for e in data:

            currentInput = {}

            # title
            videoTitle = e["name"][:-1]
            currentInput["title"] = videoTitle

            # frames list
            currentInput["frames"] = []
            for (dirPath, dirNames, fileNames) in walk(join(self.inputImagesFolder, videoTitle)):
                for fileName in fileNames:
                    currentInput["frames"].append(join(dirPath,fileName))
            currentInput["frames"] = sorted(currentInput["frames"], key=keySort)

            # labels
            currentInput["labels"] = e["annotations"]

            # number of frames & labels # not the same aaaarg
            imagesNumber = len(currentInput["frames"])
            labelsNumber = len(e["annotations"])
            currentLength = min(imagesNumber, labelsNumber)
            currentInput["labels"] = currentInput["labels"][:currentLength]
            currentInput["frames"] = currentInput["frames"][:currentLength]

            # range of frames
            currentInput["framesRange"] = (self.lastRange, self.lastRange + currentLength)

            currentInput["framesNumber"] = len(currentInput["frames"])

            self.lastRange = self.lastRange + currentLength

            self.inputsList.append(currentInput)
            self.framesNumber += currentLength

    def data(self):
        return self.inputsList

    def getFramesNumber(self):
        return self.framesNumber

    def loadImageAsArray(self, imagePath):
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        img = cv2.imread(imagePath)
        img = cv2.resize(img, (self.newSize[0], self.newSize[1]), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.flatten()
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        return img

    def sequences(self):
        sequences = []
        currentSequence = -1
        for video in self.inputsList:
            currentVideoFramesNumber = video["framesNumber"]
            for index in range (0, currentVideoFramesNumber, self.sequenceSize):
                if index+self.sequenceSize-1 >= currentVideoFramesNumber:
                    # alors il faut padder
                    sequences.append([])
                    currentSequence += 1
                    # on est à la fin d'une vidéo donc on peut faire ça :
                    sequences[currentSequence] += video["frames"][index:]
                    while len(sequences[currentSequence]) < self.sequenceSize:
                        sequences[currentSequence].append(video["frames"][-1])
                else:
                    # alors pas besoin de padder
                    sequences.append([])
                    currentSequence += 1
                    sequences[currentSequence] += video["frames"][index:index+self.sequenceSize]
        return sequences

    def labelSequences(self):
        sequences = []
        currentSequence = -1
        for video in self.inputsList:
            currentVideoFramesNumber = video["framesNumber"]
            for index in range (0, currentVideoFramesNumber, self.sequenceSize):
                if index+self.sequenceSize-1 >= currentVideoFramesNumber:
                    # alors il faut padder
                    sequences.append([])
                    currentSequence += 1
                    # on est à la fin d'une vidéo donc on peut faire ça :
                    sequences[currentSequence] += [one_hot_class(label) for label in video["labels"][index:]]
                    #sequences[currentSequence] += video["labels"][index:]
                    while len(sequences[currentSequence]) < self.sequenceSize:
                        sequences[currentSequence].append(one_hot_class(video["labels"][-1]))
                        #sequences[currentSequence].append(video["labels"][-1])
                else:
                    # alors pas besoin de padder
                    sequences.append([])
                    currentSequence += 1
                    sequences[currentSequence] += [one_hot_class(label) for label in video["labels"][index:index+self.sequenceSize]]
                    #sequences[currentSequence] += video["labels"][index:index+self.sequenceSize]
        return sequences

class SequenceLoader:

    def __init__(self, inputSequencesFolder, inputLabelsFile, sequenceSize):

        self.sequencesPath = inputSequencesFolder
        self.labelsPath = inputLabelsFile
        self.sequenceSize = sequenceSize

    def loadSequencesToMemory(self):

        featuresPath = join(self.sequencesPath, 'features')
        featuresSequencesPath = listdir(featuresPath)

        labelsPath = join(self.sequencesPath, 'labels')
        labelsSequencesPath = listdir(labelsPath)

        featureSequences = []
        labelSequences = []

        for sequencePath in featuresSequencesPath:
            sequence = np.load(join(featuresPath,sequencePath))
            featureSequences.append(sequence)

        for sequencePath in labelsSequencesPath:
            labelSequence = np.load(join(labelsPath,sequencePath))
            labelSequences.append(labelSequence)

        featureSequences = np.array(featureSequences)
        labelSequences = np.array(labelSequences)

        print('#'*20)
        print('#'*20)
        return featureSequences, labelSequences
