import subprocess
import os
from os import listdir
from os.path import isfile, join
import re
import sys

inputMoviesFolder = sys.argv[1]
# do not forget the / at the end of the inputMoviesFolder
# it has to be the folder containing the video files directly
outputFramesFolder = sys.argv[2]
# do not forget / at the end of the outputFramesFolder
# each of the video will have its own folder, and all the frames will be inside a jpg file

movieNames = [f for f in listdir(inputMoviesFolder) if isfile(join(inputMoviesFolder, f))]
# print(movieNames)

# Extract frames from movies, 1 frame every second
# /!\ Requires FFMPEG
def extractFrames(moviePath, outputPath):
    
    commandLine = 'ffmpeg -loglevel error -i "' + moviePath + '" -r 1 -f image2 "' + outputPath + '-%05d.jpg"'
    os.system(commandLine)

if __name__ == '__main__':

    for movieName in movieNames:
        print('Processing '+movieName+'...')

        movieNameParts = movieName.split('.')
        movieNameRoot = movieNameParts[0] + '/jpg/'
        outputName = movieNameParts[0]
        outputRoot = outputFramesFolder + movieNameRoot

        if not os.path.exists(outputRoot):
            os.makedirs(outputRoot)

        outputName = outputRoot + outputName
        inputName = inputMoviesFolder + movieName 
        extractFrames(inputName,outputName)
