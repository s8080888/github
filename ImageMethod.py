import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-ORC\tesseract.exe'


class ImageDetectMethod:
    def __init__(self,image):
        #self.Cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        #self.ret, self.image = self.Cap.read()
        self.image = image
        self.Blur = cv2.GaussianBlur(image, (9, 9), 9)
        self.img_canny = cv2.Canny(self.Blur, 10, 70)
        self.MidLine = 0

        self.img_Text = []
        self.img_threshold = []
        self.circles = []
        self.center = []

        self.Correct = 0
        self.TowardErr = 0
        self.TextErr = 0
        self.Loss = 0
        self.result = []

    def ShowImage(self):

        Image_list = [self.image, self.img_canny]
        Img_Name = ['image', 'canny']

        ListImg = [self.img_threshold, self.img_Text]
        list_Name = ['threshold', 'img_Text']

        for num, show_img in enumerate(Image_list):
            if show_img is not None:
                cv2.imshow(Img_Name[num], show_img)
                key = cv2.waitKey(100)

        for n,showListImg in enumerate(ListImg):
            if showListImg:
                count = len(showListImg)
                for time in range(count):
                    cv2.imshow(list_Name[n] + ' ' + str(time), showListImg[time])
                    key = cv2.waitKey(100)

        return key

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

    def FindObject(self,bounding=False):
        """
        :param bounding:是否框選物體
        """
        result = False

        threshold = cv2.cvtColor(self.Blur, cv2.COLOR_BGR2GRAY)
        _, image_threshold = cv2.threshold(threshold, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 15000:
                continue

            else:
                x, y, w, h = cv2.boundingRect(contour)
                if bounding:
                    cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                self.img_threshold.append(image_threshold[y:y+h, x:x+w])
                self.center.append(((x * 2 + w) / 2, (y * 2 + h) / 2))

                result = True

        num = len(self.img_threshold)
        for time in range(num):
            self.result.append([0,0])
        self.result.append(0)
        return result

    def FindCircle(self):

        self.circles = cv2.HoughCircles(self.img_canny, cv2.HOUGH_GRADIENT, 1, 180,
                                        param1=100, param2=20, minRadius=1, maxRadius=20)
        if self.circles is None:
            result = False
        else:
            result = True
            for i in self.circles[0, :]:
                cv2.circle(self.image, (i[0], i[1]), int(i[2]), (0, 255, 0), 2)
                cv2.circle(self.image, (i[0], i[1]), 2, (0, 0, 255), 3)

        return result

    def detectTowards(self):
        result = False
        for n, i in enumerate(self.circles[0, :]):
            k = self.SubFindMin(i[1],bool=False)
            if i[0] > self.center[k][0]:
                result = True
            else:
                result = False

        return result

    def detectText(self, x_left = -30, x_right = -55, y_top = 40, y_down = -40,mode = 0):

        """
        :param x_left:  X的下限
        :param x_right: X的上限
        :param y_top:   Y的上限
        :param y_down:  Y的下限
        :param mode: 辨識模式 OCR二值化物件
        """
        result = False

        if self.circles is None:
            return False
        for i in self.circles[0, :]:
            crop_x = int(i[0])
            crop_y = int(i[1])

            k = self.SubFindMin(crop_y,bool=False)

            TextImg = self.image[crop_y - y_top:crop_y - y_down, crop_x - x_left:crop_x - x_right]

            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
            TextImg = cv2.filter2D(TextImg, -1, kernel=kernel)
            TextImg = cv2.cvtColor(TextImg, cv2.COLOR_BGR2GRAY)
            TextImg = cv2.GaussianBlur(TextImg, (3, 3), 9)
            TextImg_Canny = cv2.Canny(TextImg, 50, 150, L2gradient=True)

            _, TextResult = cv2.threshold(TextImg_Canny, 85, 255, cv2.THRESH_BINARY)
            contours_Text, _ = cv2.findContours(TextResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.putText(self.image,str(k), (crop_x-10, crop_y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 255), 1, cv2.LINE_AA)

            if mode == 1:
                result_eng = pytesseract.image_to_string(TextResult)
                arr = result_eng.split('\n')[0:-1]
                result_eng = '\n'.join(arr)
                if len(result_eng) > 0:
                    result = True
                    self.result[k][0] += 1
                    self.img_Text.append(TextResult)
                else:
                    self.result[k][1] += 1
                    result = False
            else:
                if(len(contours_Text)) > 0:
                    result = True
                    self.result[k][0] += 1
                    self.img_Text.append(TextResult)
                else:
                    self.result[k][1] += 1
                    result = False

        print(self.result)

        return result

    def CheckResult(self,threshold = 3):
        """
        :param threshold: 檢測次數
        """
        sum = self.Correct + self.TowardErr + self.TextErr

        if sum < threshold:
            return 0

    def init(self):
        self.img_canny = None
        self.Blur = None
        self.img_Text = None
        self.img_threshold = None
        self.circles = None
        self.center = []

        self.Correct = 0
        self.TowardErr = 0
        self.TowardErr = 0
        self.Loss = 0

    def Robotis(self, id, speed,COM = "COM3"):
        ser = serial.Serial(COM, 1000000, timeout=0.5)

        ser.bytesize = serial.EIGHTBITS
        arr = []
        arr.append(0xff)
        arr.append(0xff)
        arr.append(id)
        arr.append(0x05)
        arr.append(0x03)
        arr.append(0x20)
        arr.append(speed & 255)
        arr.append(speed // 256)
        tt = 0xff - (sum(arr[2:len(arr)-1]) & 255)
        arr.append(tt)
        ser.write(arr)











