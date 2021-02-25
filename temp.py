# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import time
import numpy as np
import math

#import keyboard
import serial
import pytesseract


def image_Bulr(image,kernel_size,time):
    image = cv2.GaussianBlur(image,(kernel_size, kernel_size), time)
    return image

def Robotis(id, pos, speed):
    ser = serial.Serial("COM3",1000000,timeout=0.5)
    
    ser.bytesize = serial.EIGHTBITS
    arr = []
    arr.append(0xff)
    arr.append(0xff)
    arr.append(id)
    arr.append(0x05)
    arr.append(0x03)
    arr.append(0x20)
    # arr.append(pos & 255)
    # arr.append(pos // 255)
    arr.append(speed & 255)
    arr.append(speed // 256)
    tt = 0xff - (sum(arr[2:9]) & 255)
    arr.append(tt)
    ary = bytearray(arr)
    ser.write(arr)

    

pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-ORC\tesseract.exe'



cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_EXPOSURE,-6)
cap.set(cv2.CAP_PROP_SETTINGS,1)
cap.set(cv2.CAP_PROP_GAIN,0)
cap.set(cv2.CAP_PROP_FOCUS,10)


Loss = 0

while True:
    # Robotis(13,20,256)

    T = 0
    F = 0
    E = 0

    time_test = time.time()
    time_exe = 5
    
    _,image = cap.read()

    image = image[400:900,:1500] 
    
    cv2.line(image,(200,0),(200,500),(255,0,0),1)
    cv2.line(image,(1300,0),(1300,500),(255,0,0),1)
    
    text = np.zeros_like(image)

    Bulr_image = image_Bulr(image,9, 9)    

    image_threshold = cv2.cvtColor(Bulr_image, cv2.COLOR_BGR2GRAY)  

    ret,new_image = cv2.threshold(image_threshold,128,255,cv2.THRESH_BINARY)

    img_canny = cv2.Canny(Bulr_image,10,70)
    
    contours, _  = cv2.findContours(new_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center_object = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if cv2.contourArea(contour) < 15000:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        center_object = (x+x+w)/2

        angle = math.atan2(h,w) * (180 / math.pi)
        
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        min_rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(min_rect))


        center = ((x+w) // 2 , (y+h) // 2)
        # cv2.circle(image,(int(center_object),y),2,(0,0,255),3)            
        crop_img = image[y:y+h,x:x+w]
        
    # if ((center_object > 700) & (cent1er_object < 800)):
    if (time_exe == 5):
        #Robotis(13,24,0)
        
        for i in range(time_exe):
                
            #----------------------------   找圓   --------------------------
                
            circles = cv2.HoughCircles(img_canny,cv2.HOUGH_GRADIENT,1,180,
                                        param1=100,param2=20,minRadius=1,maxRadius=20)
            #----------------------------End 找圓--------------------------
            if circles is None:
                # print("Loss")
                Loss = Loss + 1
                cv2.imshow('x',image)
                cv2.imshow('b',img_canny)
                cv2.waitKey(1)
            
                continue
            
            else:
                for i in circles[0,:]:
                    circle_x = i[0]
                    
                    # draw the outer circle
                    cv2.circle(image,(i[0],i[1]),int(i[2]),(0,255,0),2)
                    
                    # draw the center of the circleq
                    cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)
                    
                    crop_x,crop_y = int(i[0]),int(i[1])
                    text = image[crop_y-40:crop_y+40,crop_x+30:crop_x+55]
                    
                    if not np.all(text==0): 
                        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
                        text = cv2.filter2D(text, -1, kernel=kernel)
                        dst = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
                        text = image_Bulr(dst,3, 9)
                        text_1 = cv2.Canny(text, 50, 150,L2gradient=True)
                        _, text_1 = cv2.threshold(text_1,85,255,cv2.THRESH_BINARY)
                        contours_text, _  = cv2.findContours(text_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if center_object > circle_x:
                E = E + 1
                continue
                
            if not np.all(text==0):    
                cv2.imshow('d',text)
                cv2.imshow('e',text_1)
                
                result_eng = pytesseract.image_to_string(text_1)
                arr = result_eng.split('\n')[0:-1]
                result_eng = '\n'.join(arr)
                
                # print(result_eng)
                if len(contours_text) > 0:
                # if len(result_eng) > 0:
                    T = T + 1
                    # print("True, time: %.2f" % time_now)
                    
                else:
                    F = F + 1
                    # print("False, time:%.2f" % time_now)       
            
            cv2.imshow('x',image)
            cv2.imshow('b',img_canny)
            cv2.imshow('c',new_image)
            
            x = cv2.waitKey(1)
            
            if x == ord('q'):
                print()
                print("True: ",T,", Fasle: ",F,", Loss: ",Loss)
        
                cv2.destroyAllWindows()
                cap.release()
                break
            

        if ((T <= 0) & (F <= 0) & (E <= 0)):
            
            # print("全Loss")
            continue
        
        time5 = time.time() - time_test
        print()
        # print("%.2f" % time5)
        print("\r檢測耗時：%.2f " % time5,flush=True)
        print("偵測結果： ",end="")

        # print(T,F,E,Loss)
        if E < 1:
            if T > F:
                print("朝向正確，文字正確，True")
            else:
                print("朝向正確，文字錯誤，False")
        else:
            print("朝向錯誤")
            
        
        # time.sleep(5000)

cv2.destroyAllWindows()
cap.release()
