"""
This module reads in the raw data from 4SightFocus .rawframe files
"""

import numpy as np

def read(filename):
    f = open(filename, 'rb')
    rawbytes =  f.read()
    data = np.fromstring(rawbytes, dtype=np.uint8)
    header = data[:36]
    data = data[36:]
    xsize = header[8] + 256 * header[9]
    ysize = header[12] + 256 * header[13]
    framenumber = header[0] + 258 * header[1]
    data.shape = (ysize, xsize)
    return data
