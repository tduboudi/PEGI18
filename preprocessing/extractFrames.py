import subprocess
import os
import re

movieFolder = '../../DATA_PEGI18/MEDIAEVAL17-TestSet-Data/data/'
outputFolder = '../../DATA_PEGI18/MEDIAEVAL17-TestSet-Data/frames'

movieNames = [
'MEDIAEVAL17_00.mp4',
'MEDIAEVAL17_01.mp4',
'MEDIAEVAL17_02.mp4',
'MEDIAEVAL17_03.mp4',
'MEDIAEVAL17_04.mp4',
'MEDIAEVAL17_05.mp4',
'MEDIAEVAL17_06.mp4',
'MEDIAEVAL17_07.mp4',
'MEDIAEVAL17_08.mp4',
'MEDIAEVAL17_09.mp4',
'MEDIAEVAL17_10.mp4',
'MEDIAEVAL17_11.mp4',
'MEDIAEVAL17_12.mp4',
'MEDIAEVAL17_13.mp4',
]

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
        outputRoot = outputFolder + movieNameRoot
        
        if not os.path.exists(outputRoot):
            os.makedirs(outputRoot)

        outputName = outputRoot + outputName
        inputName = movieFolder + movieName 
        extractFrames(inputName,outputName)
