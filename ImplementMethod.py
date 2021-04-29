from ImageMethod import ImageDetectMethod
import cv2
import numpy as np
import pytesseract
import time
import sys
from scipy import ndimage
import itertools
from wrapt_timeout_decorator import *


pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-ORC\tesseract.exe'

class ImplementDetectMethod:
    def __init__(self):
        self.cap = None
        self.Method = None
        self.threshold = []
        self.center = []
        self.img_Text = []
        self.ShowThreshold = None
        self.circles = None
        self.image = None
        self.time = 0
        self.CapCounter = 0
        self.bais = 0
        self.MidLine = 0

        self.result = [[0,0],[0,0]]
        self.Loss = 0


    def WebCam(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 10)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        #White Blanace 4250

    def Do(self):
        self.time = time.time()
        DetectResult = True
        self.Detect()
        for i in range(2):
            if (self.result[i][1] >= 5):
                DetectResult = False
            else:
                DetectResult = True
        result = self.result
        self.result = [[0, 0], [0, 0]]
        print(result)
        self.time = time.time()
        return DetectResult, result

    def Detect(self):
        self.UpdateData()
        self.Method.ShowImage(img_threshold=self.ShowThreshold, img_Text=self.img_Text)
        self.img_Text = []
        self.detectTowards()
        self.detectText()

        name = ["下面", "上面"]
        if self.CheckResult():
            print()
            for i in range(2):
                if np.any(self.result,axis=1)[i]:
                    if self.result[i][0] > self.result[i][1]:
                        print("%s 正確" % name[i])
                    else:
                        if (self.result[i][1] > 20):
                            print("%s 朝向錯誤" % name[i])
                        else:
                            print("%s 錯誤" % name[i])
                else:
                    print("%s 空的" % name[i])

        else:
            self.TimeOut()
            self.Detect()

    def UpdateData(self):
        ret, image = self.cap.read()

        if ret:
            self.image = image[350:850, 700:1650]
            self.image = cv2.flip(self.image, -1)

        else:
            self.TimeOut()
            self.cap.release()
            self.WebCam()
            self.UpdateData()

        self.Method = ImageDetectMethod(self.image)
        self.threshold, self.center, self.ShowThreshold = self.Method.FindObject()
        self.circles = self.Method.FindCircle()

        if self.circles is None:
            self.TimeOut()
            self.cap.release()
            self.WebCam()
            self.UpdateData()

    def SubFindNear(self, SubNum=0, XY_AxisDecider=True) -> int:
        """
        :param bool:計算X為True,反之為False
        """
        if XY_AxisDecider:
            replace = 0
        else:
            replace = 1

        transpose_list = list(list(i) for i in zip(*self.center)) #行列互換，0為X、1為Y
        minNum = min(transpose_list[replace], key=lambda c: abs(c - SubNum))
        index = transpose_list[replace].index(minNum)
        return index

    def detectTowards(self):
        for n, i in enumerate(self.circles[0, :]):
            k = self.SubFindNear(i[1], False)
            if i[0] < self.center[k][0]:
                self.result[k][1] += 25

    def detectText(self, x_left=-30, x_right=-55, y_top=25, y_down=-30, mode=0):
        """
        :param x_left:  X的下限
        :param x_right: X的上限
        :param y_top:   Y的上限
        :param y_down:  Y的下限
        :param mode: 辨識模式 OCR二值化物件
        """
        for i in self.circles[0, :]:

            crop_x = int(i[0] + self.bais)
            crop_y = int(i[1])
            Area = abs(x_left - x_right) * abs(y_top - y_down)

            k = self.SubFindNear(crop_y, False)
            Crop_TextImg = self.image[crop_y - y_top:crop_y - y_down, crop_x - x_left:crop_x - x_right]
            TextImg = cv2.GaussianBlur(Crop_TextImg, (3, 3), 5)

            # TextImg_Blur = cv2.GaussianBlur(Crop_TextImg, (3,3), 1)
            # TextImg = cv2.ximgproc.jointBilateralFilter(TextImg_Blur, TextImg, 3, 100, 1, cv2.BORDER_DEFAULT)
            # TextImg = cv2.medianBlur(TextImg, 3)

            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
            TextImg = cv2.filter2D(TextImg, -1, kernel=kernel)

            # TextImg = cv2.detailEnhance(TextImg)

            TextImg = cv2.cvtColor(TextImg, cv2.COLOR_BGR2GRAY)
            TextImg_Canny = cv2.Canny(TextImg, 30, 200, L2gradient=True)

            # TextImg_Sobel = self.Sobel(TextImg)
            # TextImg_Sobel = np.uint8(np.absolute(TextImg_Sobel))

            _, TextResult = cv2.threshold(TextImg_Canny, 55, 255, cv2.THRESH_BINARY)
            contours_Text, _ = cv2.findContours(TextResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            self.img_Text.append(TextResult)

            if mode == 1:
                result_eng = pytesseract.image_to_string(TextResult)
                arr = result_eng.split('\n')[0:-1]
                result_eng = '\n'.join(arr)

            else:
                try:
                    ContourPointNum = list(itertools.chain(*contours_Text))
                    percentage = (len(ContourPointNum) / Area) * 100

                    # if k == 1:
                    #     print("上面", end="")
                    # else:
                    #     print("下面", end="")
                    # print(": ", end="")
                    # print("%.2f" % percentage)

                    if percentage > 10:
                        self.result[k][0] += 1
                    else:
                        self.result[k][1] += 1

                except ValueError:
                    self.result[k][1] += 1

    def Sobel(self, img):
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

        absX = cv2.convertScaleAbs(x)  # 轉回uint8
        absY = cv2.convertScaleAbs(y)

        img_Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        img_Sobel = img_Sobel.astype('uint8')
        return img_Sobel

    def CheckResult(self,threshold=5):
        """
        :param threshold: 檢測次數
        """

        sum = 0
        for i in range(len(self.threshold)):
            sum += self.result[i][0]
            sum += self.result[i][1]
            if sum < threshold:
                return False
            else:
                sum = 0
                continue

        return True

    def TimeOut(self, limitTime=3):
        nowTime = time.time()
        if (nowTime - self.time) > limitTime:
            raise TimeoutError("TimeOut")
