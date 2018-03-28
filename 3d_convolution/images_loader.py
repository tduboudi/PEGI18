#!/usr/bin/python3.6

import os
from os.path import isfile, join, isdir
from os import walk
import numpy as np
import json
import cv2

def one_hot(labels):
    return np.eye(np.max(labels) + 1)[labels]

class ImageLoader:
    def __init__(self, inputImagesFolder, inputLabelsFile, sequenceSize, image_shape):
        self.inputImagesFolder = inputImagesFolder
        self.inputLabelsFile = inputLabelsFile
        self.sequenceSize = sequenceSize
        self.image_shape = image_shape

    def loadImageAsArray(self, imagePath):
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        img = cv2.imread(imagePath)
        img = cv2.resize(img, (self.image_shape[0], self.image_shape[1]), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    def load(self):
        with open(self.inputLabelsFile) as f:
            data = json.loads(f.read())

        sequences = []
        labels    = []

        videoNames = [f for f in os.listdir(self.inputImagesFolder) if isdir(join(self.inputImagesFolder, f))]
        videoNames = [videoNames[1]]

        for videoName in videoNames:
            folder = join(self.inputImagesFolder, videoName, 'jpg')
            images = [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f))];
            annotations = []

            for videoLabels in data:
                if videoLabels['name'] == videoName:
                    annotations = videoLabels['annotations']

            for i in range(len(images) - self.sequenceSize):
                frames = []

                for j in range(self.sequenceSize):
                    frames.append(self.loadImageAsArray(images[i+j]))

                sequences.append(frames)
                print(i)
                labels.append(annotations[i+self.sequenceSize])

        return np.array(sequences), one_hot(np.array(labels))
