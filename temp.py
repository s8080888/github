# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time
import numpy as np

from ImplementMethod import ImplementDetectMethod
from SerialPortAccept import SerialPortTerminal

from multiprocessing import Pool, TimeoutError

def Detect(switch, Method):
    if switch:
        Result, ResultList = Method.Do()
        return [Result, ResultList]


if __name__ == '__main__':
    TimeMean = []
    Method = ImplementDetectMethod()
    Serial = SerialPortTerminal()
    Method.WebCam()
    while True:
        try:
                Switch = Serial.ReadCommand()
                time_start = time.time()
                Detect(Switch, Method)
                time_end = time.time() - time_start
                print("耗費時間共 %.2f " % time_end)
                TimeMean.append(time_end)

        except KeyboardInterrupt:
            Serial.EndAndClose()

        except TimeoutError:
            print("TimeOut")
            continue

        # except Exception as e:
        #     print(e)
        #     Serial.EndAndClose()

        # finally:
        #     Serial.EndAndClose()
        #     print("總共執行次數：%d" % len(TimeMean))
        #     print("平均執行時間：%.2f" % np.mean(TimeMean))
