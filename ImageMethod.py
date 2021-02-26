import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-ORC\tesseract.exe'


class ImageDetectMethod:

    def __init__(self,image):
        self.image = image
        self.img_canny = None
        self.Blur = None
        self.img_Text = None
        self.img_threshold = None
        self.circles = None
        self.center_x = 0
        self.Image_list = [self.image, self.img_threshold, self.img_canny, self.img_Text]


    def ShowImage(self):
        Img_Name = ['image', 'threshold', 'canny', 'img_Text']

        for num, show_img in enumerate(self.Image_list):
            if show_img is not None:
                cv2.imshow(Img_Name[num], show_img)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    cv2.destroyAllWindows()

    def FindObject(self,bounding=False):
        """
        :param bounding:是否框選物體
        """
        result = False

        self.Blur = cv2.GaussianBlur(self.image, (9, 9), 9)
        threshold = cv2.cvtColor(self.Blur, cv2.COLOR_BGR2GRAY)

        _, self.img_threshold = cv2.threshold(threshold, 128, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(self.img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 15000:
                result = False
            else:
                if(bounding):
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                x, y, w, h = cv2.boundingRect(contour)

                self.center_x = (x * 2 + w) / 2

                result = True

        return result,self.img_threshold

    def FindCircle(self):
        self.img_canny = cv2.Canny(self.Blur, 10, 70)

        self.circles = cv2.HoughCircles(self.img_canny, cv2.HOUGH_GRADIENT, 1, 180,
                                        param1=100, param2=20, minRadius=1, maxRadius=20)

        for i in self.circles[0,:]:

            cv2.circle(self.image,(i[0],i[1]),int(i[2]),(0,255,0),2)

            cv2.circle(self.image,(i[0],i[1]),2,(0,0,255),3)

        if self.circles is None:
            result = False
        else:
            result = True
        return result,self.img_canny

    def detectTowards(self):
        for i in self.circles[0,:]:
            if i[0] > self.center_x:
                return True
            else:
                return False



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

        self.img_Text = self.image[crop_y - y_top:crop_y - y_down, crop_x - x_left:crop_x - x_right]

        if not np.all(self.img_Text==0):
            TextImg = self.img_Text

            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
            TextImg = cv2.filter2D(TextImg, -1, kernel=kernel)
            TextImg = cv2.cvtColor(TextImg, cv2.COLOR_BGR2GRAY)
            TextImg = cv2.GaussianBlur(TextImg, (3, 3), 9)
            TextImg_Canny = cv2.Canny(TextImg, 50, 150, L2gradient=True)

            _, TextResult = cv2.threshold(TextImg_Canny, 85, 255, cv2.THRESH_BINARY)
            contours_Text, _ = cv2.findContours(TextResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            self.img_Text = contours_Text

            if mode < 1:
                result_eng = pytesseract.image_to_string(TextResult)
                arr = result_eng.split('\n')[0:-1]
                result_eng = '\n'.join(arr)

                if len(result_eng) > 0:
                    result = True
                else:
                    result = False

            else:

                if(len(contours_Text)) > 0:
                    result = True
                else:
                    result = False

        return result,self.img_Text












