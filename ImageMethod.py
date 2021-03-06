import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-ORC\tesseract.exe'

class ImageDetectMethod:

    def __init__(self, image):
        self.image = image
        self.img_canny = None
        self.circles = []

    def ShowImage(self, img_threshold=None, img_Text=None):

        Image_list = [self.image, self.img_canny, img_threshold]
        Img_Name = ['image', 'canny', 'threshold']

        ListImg = [img_Text]
        list_Name = ['img_Text']

        for num, show_img in enumerate(Image_list):
            if show_img is not None:
                cv2.imshow(Img_Name[num], show_img)
                cv2.waitKey(1)

        for n, showListImg in enumerate(ListImg):
            if showListImg:
                count = len(showListImg)
                for time in range(count):
                    cv2.imshow(list_Name[n] + ' ' + str(time), showListImg[time])
                    cv2.waitKey(1)

    def FindObject(self,bounding=False):
        """
        :param bounding:是否框選物體
        """
        img_threshold = []
        resultList = []
        center = []

        image = cv2.GaussianBlur(self.image, (3,3), 3)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([38, 63, 80])
        upper_blue = np.array([85, 255, 255])

        threshold = cv2.inRange(hsv, lower_blue, upper_blue)
        threshold = cv2.bitwise_not(threshold)

        kernel = np.ones((3, 3), np.uint8)
        threshold = cv2.erode(threshold, kernel, iterations=5)
        threshold = cv2.dilate(threshold, kernel, iterations=5)

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if (area < 35000) or (area > 100000):
                continue
            else:
                x, y, w, h = cv2.boundingRect(contour)
                if bounding:
                    cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                img_threshold.append(threshold[y:y+h, x:x+w])
                center.append(((x * 2 + w) / 2, (y * 2 + h) / 2))

        return img_threshold, center, threshold

    def Image_Fouriertransform(self, image):
        radius = 15

        # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
        fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

        # 对fft进行中心化，生成的dshift仍然是一个三维数组
        dshift = np.fft.fftshift(fft)

        # 得到中心像素
        rows, cols = image.shape[:2]
        mid_row, mid_col = int(rows / 2), int(cols / 2)

        # 构建ButterWorth高通滤波掩模
        mask = np.ones((rows, cols, 2), np.float32)
        mask[mid_row - radius:mid_row + radius, mid_col - radius:mid_col + radius] = 0

        # 给傅里叶变换结果乘掩模
        fft_filtering = dshift * mask

        # 傅里叶逆变换
        ishift = np.fft.ifftshift(fft_filtering)
        image_filtering = cv2.idft(ishift)
        image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])

        # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
        cv2.normalize(image_filtering, image_filtering, 0, 255, cv2.NORM_MINMAX)
        image_filtering = image_filtering.astype('uint8')

        return image_filtering

    def FindCircle(self):
        img = cv2.cvtColor(self.image[:, :], cv2.COLOR_BGR2GRAY)
        self.img_canny = self.Image_Fouriertransform(img)
        self.circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 180,
                                        param1=100, param2=20, minRadius=10, maxRadius=15)

        if self.circles is None:
            pass
        else:
            for i in self.circles[0, :]:
                cv2.circle(self.image, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
                cv2.circle(self.image, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

        return self.circles






