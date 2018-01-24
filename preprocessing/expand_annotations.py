#!/usr/bin/python3.6

import re
import json
import datetime

filename = 'currated.json'
output   = 'expanded.json'

def convertTime(hour):
    return 3600*hour.hour + 60*hour.minute + hour.second

def fear(sequences, end):

    endTime  = datetime.datetime.strptime(end, '%H:%M:%S').time()
    endStamp = convertTime(endTime) + 1

    annotations = endStamp * [0]

    for sequence in sequences:
        begin = convertTime(datetime.datetime.strptime(sequence['beginning'], '%H:%M:%S').time())
        end   = convertTime(datetime.datetime.strptime(sequence['end'],   '%H:%M:%S').time())

        for i in range(begin, end):
            annotations[i] = 1

    return annotations


file = open(filename, "r")
data = json.load(file)
expanded = []

for film in data:
    current = {'name': film['name'], 'annotations': fear(film['sequences'], film['end'])}
    expanded.append(current)

file.close()

with open(output, 'w') as file:
    json.dump(expanded, file)
