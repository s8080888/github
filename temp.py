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

from multiprocessing import Pool, TimeoutError
from os import getpid

TimeMean = []
Method = ImplementDetectMethod()


def Detect(switch, Method):
    if switch:
        Result, ResultList = Method.Do()
        return [Result, ResultList]


if __name__ == '__main__':
    TimeMean = []
    Method = ImplementDetectMethod()
    Serial = SerialPortTerminal()
    with Pool(processes=1) as Pool:
        try:

            while True:
                Switch = Serial.ReadCommand()
                time_start = time.time()
                res = Pool.apply_async(Detect, (Switch, Method))
                Result = res.get(30)
                time_end = time.time() - time_start
                print("耗費時間共 %.2f " % time_end)
                TimeMean.append(time_end)
                
        except KeyboardInterrupt:
            Serial.EndAndClose()
        
        except TimeoutError:
            print("TimeOut")
        
        except Exception as e:
            print(e)
            Serial.EndAndClose()
    
        finally:
            Serial.EndAndClose()
            print("總共執行次數：%d" % len(TimeMean))
            print("平均執行時間：%.2f" % np.mean(TimeMean))
