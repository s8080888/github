# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:31:08 2021

@author: lab
"""

import serial
import keyboard
from time import sleep

ser = serial.Serial('COM7',9600)

print('案K執行')
# try:
while True:
    data_raw = ser.readline()
    print(int(data_raw))
    if((int(data_raw)) == 1):
        ser.write(b'2')
        sleep(0.5)

    # if keyboard.is_pressed('k'):
    #     print("傳送2")
    #     ser.write(b'2')
    #     sleep(0.5)
    
    if keyboard.is_pressed('ESC'):
        print('結束')
        ser.close()
        break
# except:
#     print("error")
#     ser.close()


