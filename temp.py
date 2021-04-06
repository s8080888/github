# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import time
import numpy as np
import math

import keyboard
import serial
import pytesseract
from ImageMethod import ImageDetectMethod
from ImplementMethod import ImplementDetectMethod
from SerialPortAccept import SerialPortTerminal


TimeMean = []
Method = ImplementDetectMethod()
Method.WebCam()
Serial = SerialPortTerminal()
try:
    while True:
        KeyWord = Serial.ReadCommand()
        print(KeyWord)
        
        init = np.zeros((100, 100))
        cv2.imshow('a', init)
        EndButton = cv2.waitKey(5)
        
        if EndButton == ord('q'):
            break
        if(KeyWord):
            time_start = time.time()
            Result = Method.Do()
            time_end = time.time() - time_start
            print("耗費時間共 %.2f " % time_end)
            TimeMean.append(time_end)
            print(Result)
            if Result:
                Serial.SendResult()

except Exception as e:
    print(e)
    Serial.EndAndClose()
        


Serial.EndAndClose()
cv2.destroyAllWindows()
print(len(TimeMean))
print(np.mean(TimeMean,axis=0))


#h 35:97 s 27:180 v 0 190
# h 36:99 s 86:255 v 0:255