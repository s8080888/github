# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:31:08 2021

@author: lab
"""

import serial
import keyboard
from time import sleep

class SerialPortTerminal:
    def __init__(self):
        self.ser = serial.Serial('COM7',9600)

    def ReadCommand(self):
        data_raw = self.ser.readline()
        result = lambda JudgMent: True if JudgMent > 0 else False
        return result(int(data_raw))

        
    def SendResult(self):
        self.ser.write(b'2')
        sleep(0.5)

    def EndAndClose(self):
        self.ser.close()