"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np
import os.path
from images_loader import ImageLoader
from extractor import Extractor
from tqdm import tqdm

from parameters import param

def getFrameNumber(string):
    return string[-9:-4]

def getVideoName(string):
    splitted = string.split('/')
    name_and_number = splitted[-1]
    name = name_and_number[:-10]
    return name

# get the model.
model = Extractor()

################################################################################

inputImagesFolder = param['inputTestImages']
inputLabelsFile = param['inputTestLabels']
newSize = (0,0) # not important
batchSize = 32 # not important
sequenceSize = param['seq_length']
testingImageLoader = ImageLoader(inputImagesFolder, inputLabelsFile, newSize, batchSize, sequenceSize)
imageSequences = testingImageLoader.sequences()
labelSequences = testingImageLoader.labelSequences()

# Loop through testing data.
pbar = tqdm(total=len(imageSequences))
count = 0
for sequence in imageSequences:

    lastImageNumber = getFrameNumber(sequence[-1])
    videoName = getVideoName(sequence[-1])
    # Get the path to the sequence for this video.
    path = os.path.join(param['outputTestSequences'], videoName + '-' + \
        '-features' + '-' + lastImageNumber)  # numpy will auto-append .npy

    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue
    
    # Now loop through and extract features to build the sequence.
    featureSequence = []
    for image in sequence:
        features = model.extract(image)
        featureSequence.append(features)
    
    # update the progress bar
    pbar.update(1)
    
    # Save the sequence.
    np.save(path, featureSequence)


    path = os.path.join(param['outputTestLabels'], videoName + '-' + \
        '-features' + '-' + lastImageNumber)  # numpy will auto-append .npy

    np.save(path, labelSequences[count])

    count += 1

pbar.close()



################################################################################
################################################################################
################################################################################


inputImagesFolder = param['inputTrainImages']
inputLabelsFile = param['inputTrainLabels']
newSize = (0,0) # not important
batchSize = 32 # not important
sequenceSize = param['seq_length']
trainingImageLoader = ImageLoader(inputImagesFolder, inputLabelsFile, newSize, batchSize, sequenceSize)
imageSequences = trainingImageLoader.sequences()
labelSequences = trainingImageLoader.labelSequences()

# Loop through training data.
pbar = tqdm(total=len(imageSequences))
count = 0
for sequence in imageSequences:

    lastImageNumber = getFrameNumber(sequence[-1])
    videoName = getVideoName(sequence[-1])
    # Get the path to the sequence for this video.
    path = os.path.join(param['outputTrainSequences'], videoName + '-' + \
        '-features' + '-' + lastImageNumber)  # numpy will auto-append .npy

    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue
    
    # Now loop through and extract features to build the sequence.
    featureSequence = []
    for image in sequence:
        features = model.extract(image)
        featureSequence.append(features)
    
    # update the progress bar
    pbar.update(1)
    
    # Save the sequence.
    np.save(path, featureSequence)

    path = os.path.join(param['outputTrainLabels'], videoName + '-' + \
        '-features' + '-' + lastImageNumber)  # numpy will auto-append .npy

    np.save(path, labelSequences[count])

    count += 1

pbar.close()
