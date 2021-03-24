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

k = []
Method = ImplementDetectMethod()
Method.WebCam()

while True:
    time_start = time.time()
    Method.Do()
    time_end = time.time() - time_start
    print("耗費時間共 %.2f " % time_end)
    k.append(time_end)
    init = np.zeros((100, 100))

    cv2.imshow('a', init)
    w = cv2.waitKey(5)
    if w == ord('q'):
        break


print(len(k))
print(np.mean(k,axis=0))


#h 35:97 s 27:180 v 0 190
# h 36:99 s 86:255 v 0:255