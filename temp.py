# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import time
import numpy as np
import math

# import keyboard
import serial
import pytesseract
from ImageMethod import ImageDetectMethod
from ImplementMethod import ImplementDetectMethod


while True:
    time_start = time.time()
    Method = ImplementDetectMethod()
    Method.Do()
    time_end = time.time() - time_start
    print("耗費時間共 %.2f " % time_end)


#h 35:97 s 27:180 v 0 190