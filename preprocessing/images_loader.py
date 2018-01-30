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


class ImageLoader:

    def __init__(self,inputImagesFolder, inputLabelsFile, newSize, batchSize):

        self.inputImagesFolder = inputImagesFolder
        self.inputLabelsFile = inputLabelsFile
        self.newSize = newSize
        self.batchSize = batchSize

        self.inputsList = []
        self.framesNumber = 0
        self.lastRange = 0

        self.Cursor = 0
        self.currentEpochOrder = []

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
            self.lastRange = self.lastRange + currentLength

            self.inputsList.append(currentInput)
            self.framesNumber += currentLength

        self.createEpochOrder()

    def getFramesNumber(self):
        return self.framesNumber

    def load_image(self, imagePath):
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        img = cv2.imread(imagePath)
        img = cv2.resize(img, (self.newSize[0], self.newSize[1]), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.flatten()
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        return img

    def getImagePath(self, index):
        for video in self.inputsList:
            if index >= video["framesRange"][0] and index < video["framesRange"][1]:
                # on est dans la bonne video
                wantedImagePath = video["frames"][index-video["framesRange"][0]]
                return wantedImagePath

    def getLabel(self, index):
        for video in self.inputsList:
            if index >= video["framesRange"][0] and index < video["framesRange"][1]:
                # on est dans la bonne video
                wantedLabelPath = video["labels"][index-video["framesRange"][0]]
                return wantedLabelPath

    def printWhatsInside(self):
        for video in self.inputsList:
            print(video["title"])
            print(video["framesRange"])
            count = 0
            for frame in video["frames"]:
                print(frame, video["labels"][count])
                count += 1

    def createEpochOrder(self):
        order = np.arange(self.framesNumber)
        np.random.shuffle(order)
        self.currentEpochOrder = order

    def next(self):
        if self.Cursor < self.framesNumber:
            imageBatch = []
            labelBatch = []
            startingCursor = self.Cursor
            for i in range (min(self.batchSize, self.framesNumber - startingCursor)):
                imagePath = self.getImagePath(self.currentEpochOrder[self.Cursor])
                label = self.getLabel(self.currentEpochOrder[self.Cursor])
                imageNumpy = self.load_image(imagePath)
                imageBatch.append(imageNumpy)
                labelBatch.append(label)
                self.Cursor += 1
            imageBatch = np.array(imageBatch)
            labelBatch = np.array(labelBatch, dtype=np.float32)
            return (imageBatch, labelBatch)
        else:
            print("Starting new epoch")
            self.createEpochOrder()
            self.Cursor = 0


test = ImageLoader("../../PEGI18_DATA/FRAMES/Training/", "./training_expanded.json", (640,480), 32)
print(test.printWhatsInside())
for i in range (5000):
    print("-------------------")
    print(i)
    print(test.next())
