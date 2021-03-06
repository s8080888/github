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
    Method = ImplementDetectMethod()
    Method.Do()


