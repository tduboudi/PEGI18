#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### not useful for now ###
### extractFrames.py does the work ###


import numpy as np
import cv2
import math
import json
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)

def preprocess(input_file,output_file=None):
    '''This function is used to preprocess a video file into the desired format'''

    logger.debug("Entering function preprocess")

    logger.debug("Checking if input_file exists")
    if not os.path.isfile(input_file):
        logger.critical("Input file does not exist. Exiting.")
        quit()

    logger.debug("Checking if output_file was informed")
    if output_file is None:
        logger.info("Output file does not exist. Using default output file.")
        output_file = os.path.splitext(input_file)[0] + "_output"


    logger.debug("Opening video file")
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        logger.critical("could not open : %s. Exiting." % (input_file))
        quit()

    logger.debug("Defining video attributes")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    logger.info("length : %s" % length)
    logger.info("width : %s" % width)
    logger.info("height : %s" % height)
    logger.info("fps : %s" % fps)

    logger.debug("initializing empty variables.")
    mean_frame = np.zeros((length, width,3))
    count = 0
    last_ten_frames = []


    with open(output_file, 'wb') as outfile:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                if count < 10:
                    last_ten_frames.append(frame)
                    mean_frame = np.floor(np.mean(last_ten_frames, axis=0)).astype(np.uint8)
                else:
                    last_ten_frames.append(frame)
                    del last_ten_frames[0]
                    mean_frame = np.floor(np.mean(last_ten_frames, axis=0)).astype(np.uint8)

                logger.debug("Count : %s", count)

                # np.save(outfile, mean_frame)

                # cv2.imshow('Trailer Evil Dead 2', mean_frame)
                # frames_array.append(mean_frame)
                count += 1
            else:
                logger.warn("Ret is false")
                break
            if cv2.waitKey(41) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    preprocess('evil_dead_2_trailer.mkv')
