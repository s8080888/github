# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import time
import numpy as np

from ImplementMethod import ImplementDetectMethod
from SerialPortAccept import SerialPortTerminal

import multiprocessing
from os import getpid

TimeMean = []
Method = ImplementDetectMethod()
Method.WebCam()
Serial = SerialPortTerminal()

def Detect(switch):
    print("In")
    if switch:

        time_start = time.time()
        Result = Method.Do()
        time_end = time.time() - time_start
        print("耗費時間共 %.2f " % time_end)
        TimeMean.append(time_end)
        print(Result)
        if Result:
            Serial.SendResult()
        return Result


if __name__ == '__main__':
        try:
            while True:
    
                Switch = Serial.ReadCommand()
                Detect(Switch)

        except KeyboardInterrupt:
            Serial.EndAndClose()
        
        except multiprocessing.TimeoutError:
            print("TimeOut")
        
        except Exception as e:
            print(e)
    
        finally:
            Serial.EndAndClose()
            cv2.destroyAllWindows()

# from multiprocessing import Pool, TimeoutError
# import time
 
 
# def f(test):
#     if test:
#         for i in range(10):
#             print(i)
        
#         return i
#     return 1
 
# # 因 window spawn 的緣故
# # 必須在 __name__ == '__main__' 之內執行    
# if __name__ == '__main__':
#     # start 4 worker processes
#     with Pool(processes=4) as pool:
#         A = True
#         # 建立 child process 並執行
#         resA = pool.apply_async(f, (A,))
 
#         try:
#             print(resA.get(timeout=1))
#         except TimeoutError:
#             print("得 resB 值超時")
            
#         print("pool 此時仍可使用")
        
#     # 已跳出 with，故 pool 已被 close
#     print("pool 已不可使用")
#     # 得到回傳值
#     print('resA', resA.get(timeout=1)) # resA 400