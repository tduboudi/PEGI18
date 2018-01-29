#!/usr/bin/python3.6

from os import listdir
from os.path import isfile, join
from os import walk
from os import mkdir
import json
import sys
from PIL import Image

def keySort(string):
    return string[-9:-4]


class ImageLoader:

    def __init__(self,inputImagesFolder, inputLabelsFile, newSize):
        self.inputImagesFolder = inputImagesFolder
        self.inputLabelsFile = inputLabelsFile
        self.newSize = newSize
        self.inputsList = []
        self.framesNumber = 0

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

            # number of frames
            currentInput["framesNumber"] = len(e["annotations"])

            self.inputsList.append(currentInput)
            self.framesNumber += currentInput["framesNumber"])

    def next(self):
        # qu'est ce qu'on fait là ?





test = ImageLoader("../../PEGI18_DATA/FRAMES/Training/", "./training_expanded.json", (1920,1080))


# def resizeImagesInFolder(inputImagesFolder)
