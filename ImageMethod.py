import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-ORC\tesseract.exe'

class ImageDetectMethod:

    def __init__(self,image):
        self.image = image
        self.Blur = cv2.GaussianBlur(image, (9, 9), 9)
        self.img_canny = cv2.Canny(self.Blur, 3, 45)

        self.circles = []
        self.center = []

    def ShowImage(self, img_threshold=None, img_Text=None):

        Image_list = [self.image, self.img_canny]
        Img_Name = ['image', 'canny']

        ListImg = [img_threshold, img_Text]
        list_Name = ['threshold', 'img_Text']

        for num, show_img in enumerate(Image_list):
            if show_img is not None:
                cv2.imshow(Img_Name[num], show_img)
                key = cv2.waitKey(1)

        for n,showListImg in enumerate(ListImg):
            if showListImg:
                count = len(showListImg)
                for time in range(count):
                    cv2.imshow(list_Name[n] + ' ' + str(time), showListImg[time])
                    key = cv2.waitKey(1)

                cv2.destroyWindow(list_Name[n] + ' ' + str(time))

        return key

    def FindObject(self,bounding=False):
        """
        :param bounding:是否框選物體
        """
        img_threshold = []
        resultList = []

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

                img_threshold.append(image_threshold[y:y+h, x:x+w])
                self.center.append(((x * 2 + w) / 2, (y * 2 + h) / 2))

        num = len(img_threshold)
        for time in range(num):
            resultList.append([0,0])
        resultList.append(0)

        return img_threshold, self.center, resultList

    def FindCircle(self):
        self.circles = cv2.HoughCircles(self.img_canny, cv2.HOUGH_GRADIENT, 1, 180,
                                        param1=100, param2=20, minRadius=1, maxRadius=20)
        k = 0
        if self.circles is None:
            return self.circles
        else:
            for i in self.circles[0, :]:

                cv2.circle(self.image, (i[0], i[1]), int(i[2]), (0, 255, 0), 2)
                cv2.circle(self.image, (i[0], i[1]), 2, (0, 0, 255), 3)

                k = k + 1
        return self.circles

    def detectTowards(self):
        result = False
        for n, i in enumerate(self.circles[0, :]):
            k = self.SubFindMin(i[1],bool=False)
            if i[0] > self.center[k][0]:
                result = True
            else:
                result = False

        return result

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











