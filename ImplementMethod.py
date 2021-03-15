from ImageMethod import ImageDetectMethod
import cv2
import numpy as np
import pytesseract
import time
import sys
from scipy import ndimage

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
        self.k = 0
        self.bais = 0

        self.result = [[0,0],[0,0]]
        self.Loss = 0

    def WebCam(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, -4)
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 10)

    def Do(self):
        time_strat = time.time()
        self.Detect()
        print(self.result)
        self.result = [[0,0],[0,0],0]
        print()
        time_end = time.time() - time_strat
        print("耗費時間共 %.2f " % time_end)

    def Detect(self):
        self.UpdateData()
        self.k = self.Method.ShowImage(img_threshold=self.ShowThreshold, img_Text=self.img_Text)

        if self.circles is None:
            self.Loss += 1
            self.cap.release()
            self.WebCam()
            self.UpdateData()

        self.detectText()

        if self.CheckResult():
            for i in range(2):
                if self.result[i][0] > self.result[i][1]:
                    print("第 %d 正確" % i,end=" ")
                else:
                    print("第 %d 錯誤" % i,end=" ")

        else:
            self.Detect()

    def UpdateData(self):
        ret, image = self.cap.read()

        if self.cap.isOpened():
            # image = cv2.flip(image, -1)
            self.image = image[450:850, 600:1500]
        else:
            pass
        self.entity(self.image)

    def entity(self,image):
        self.Method = ImageDetectMethod(image)
        self.threshold, self.center, self.ShowThreshold = self.Method.FindObject()
        self.circles, self.bais = self.Method.FindCircle()

    def SubFindMin(self, SubNum = 0, bool=True):
        """
        :param bool:計算X為True,反之為False
        """
        if bool:
            replace = 0
        else:
            replace = 1

        transpose_list = list(list(i) for i in zip(*self.center)) #行列互換，0為X、1為Y
        minNum = min(transpose_list[replace], key=lambda c: abs(c - SubNum))
        index = transpose_list[replace].index(minNum)
        return index

    def detectText(self, x_left=-30, x_right=-55, y_top=40, y_down=-40, mode=0):

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

            k = self.SubFindMin(crop_y,bool=False)
            TextImg = self.image[crop_y - y_top:crop_y - y_down, crop_x - x_left:crop_x - x_right]
            # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
            # dst = cv2.filter2D(TextImg, -1, kernel=kernel)
            # dst = cv2.GaussianBlur(dst, (3,3), 1)
            TextImg_Canny = cv2.Canny(TextImg, 30, 150, L2gradient=True)
            _, TextResult = cv2.threshold(TextImg_Canny, 55, 255, cv2.THRESH_BINARY)
            contours_Text, _ = cv2.findContours(TextResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if mode == 1:
                result_eng = pytesseract.image_to_string(TextResult)
                arr = result_eng.split('\n')[0:-1]
                result_eng = '\n'.join(arr)
                if len(result_eng) > 0:
                    self.result[k][0] += 1
                    self.img_Text.append(TextResult)
                else:
                    self.result[k][1] += 1
            else:
                if(len(contours_Text)) > 0:
                    self.result[k][0] += 1

                    self.img_Text.append(TextResult)
                else:
                    self.result[k][1] += 1

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
                continue
        return True