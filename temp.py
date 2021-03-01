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

    T = 0
    F = 0
    E = 0

    time_test = time.time()

    _, image = cap.read()

    image = image[400:900, :1500]

    cv2.line(image, (200, 0), (200, 500), (255, 0, 0), 1)
    cv2.line(image, (1300, 0), (1300, 500), (255, 0, 0), 1)

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
        Loss = Loss + 1
        TestMethod.ShowImage()

        continue

    if Toward:
        result = TestMethod.detectText()
    else:
        E = E + 1
        TestMethod.ShowImage()

        continue

    if result:
        T = T + 1
    else:
        F = F + 1
        TestMethod.ShowImage()

        continue

    sum = T + F + E

    if ((T <= 0) & (F <= 0) & (E <= 0)):
        TestMethod.ShowImage()
        continue

    x = TestMethod.ShowImage()

    if x == ord('q'):
        print()
        print("True: ", T, ", Fasle: ", F, ", Loss: ", Loss)

        cv2.destroyAllWindows()
        cap.release()
        break

        time5 = time.time() - time_test
        print()
        print("\r檢測耗時：%.2f " % time5, flush=True)
        print("偵測結果： ", end="")

        if E < 1:
            if T > F:
                print("朝向正確，文字正確，True")
            else:
                print("朝向正確，文字錯誤，False")
        else:
            print("朝向錯誤")

cv2.destroyAllWindows()
cap.release()