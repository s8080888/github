# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:31:08 2021

@author: lab
"""
import serial
import keyboard
from time import sleep
from serial.tools.list_ports import comports
import re


class SerialPortTerminal:
    def __init__(self):
        DetectAllCOMPort = comports()
        for COMPORT in DetectAllCOMPort:
            if (re.match('Arduino Uno \(COM[0-9]\)',COMPORT[1]) != None):
                ConnectNumber = str(COMPORT[0])
                
        self.ser = serial.Serial(ConnectNumber, 9600)

    def ReadCommand(self):
        data_raw = self.ser.readline()
        result = lambda JudgMent: True if JudgMent > 0 else False
        return result(int(data_raw))

    def SendResult(self):
        self.ser.write(b'2')
        sleep(0.5)

    def EndAndClose(self):
        self.ser.close()

X = comports()

print(X.description)