#!/usr/bin/python3.6

import re
import json

filename = 'annotations.txt'
output   = 'currated.json'

file = open(filename, "r")
data = []

current = {'sequences': []}

for line in file:
    if re.match(r'^\[INFO\]', line):   # Beginning match [INFO]
        data.append(current)
        current = {'sequences': []}
        continue

    if not re.match(r'^\[F\]', line): # Beginning match [F]
        current['name'] = line
        continue

    beginning = re.search(r'^\[F\](.*) - Start', line).group(1)
    end       = re.search(r'\|\[F\](.*) - Stop', line).group(1)

    current['sequences'].append({'beginning': beginning, 'end': end})

file.close()

with open(output, 'w') as file:
    json.dump(data, file)
