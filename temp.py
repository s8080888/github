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

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_SETTINGS, 1)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_FOCUS, 10)

Loss = 0

while True:

    # Robotis(13,20,256)


    TestMethod = ImageDetectMethod(image)
    Object = TestMethod.FindObject()

    if Object:
        circle = TestMethod.FindCircle()
    else:
        TestMethod.ShowImage()
        continue

    if circle:
        Toward = TestMethod.detectTowards()
    else:
        TestMethod.ShowImage()
        continue

    if Toward:
        result = TestMethod.detectText()
    else:
        TestMethod.ShowImage()
        continue

    if result:
        T = T + 1
    else:
        F = F + 1
        TestMethod.ShowImage()
        continue

    x = TestMethod.ShowImage()

    if x == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        cost_time = time.time() - time_test

        print()
        print("\r檢測耗時：%.2f " % cost_time, flush=True)

        break

cv2.destroyAllWindows()
cap.release()