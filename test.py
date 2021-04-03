import numpy as np
from ImplementMethod import ImplementDetectMethod
from ImageMethod import ImageDetectMethod
import cv2
import math
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_SETTINGS, 1)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_FOCUS, 10)



def Sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 轉回uint8
    absY = cv2.convertScaleAbs(y)

    img_Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    img_Sobel = img_Sobel.astype('uint8')
    return img_Sobel

while True:
    mode = 0
    w = [0, 0]
    ret, image = cap.read()

    image = image[200:700, 500:1600]
    image = cv2.flip(image, -1)
    image = cv2.detailEnhance(image)
    img_canny = cv2.Canny(image,30,150)

    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    img_Laplacian = cv2.convertScaleAbs(gray_lap)

    img_Sobel = Sobel(img)

    TextResult = np.zeros((40,40))

    img1 = image.copy()
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    img1 = cv2.filter2D(img1, -1, kernel=kernel)

    radius = 5
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
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
    image_filtering2 = image_filtering.astype('uint8')

    circles = cv2.HoughCircles(img_Sobel, cv2.HOUGH_GRADIENT, 1, 180,
                                    param1=110, param2=25, minRadius=10, maxRadius=15)

    TextImg_Canny = np.zeros((20,20))

    if circles is None:
        pass
    else:
        print(len(circles[0, :]))
        for i in circles[0, :]:
            cv2.circle(image, (int(i[0]), int(i[1])), int(int(i[2])), (0, 255, 0), 2)
            cv2.circle(image, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

            w[0] = int(i[0])
            w[1] = int(i[1])

    if circles is not None:
        x_left = -30
        x_right = -55
        y_top = 40
        y_down = -40
        textImg = image[w[1] - y_top:w[1] - y_down, w[0] - x_left: w[0] - x_right]
        textImg = cv2.GaussianBlur(textImg, (3, 3), 1)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
        # textImg = cv2.filter2D(textImg, -1, kernel=kernel)
        try:
            textImg = cv2.detailEnhance(textImg)
            textImg = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)
            TextImg_Canny = cv2.Canny(textImg, 50, 150, L2gradient=True)
            _, TextResult = cv2.threshold(TextImg_Canny, 85, 255, cv2.THRESH_BINARY)
            Text, _ = cv2.findContours(TextResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        except:
            print("pass")
            pass
        try:
            print(len(Text[1]), end=" ")
            print(" ok", end=" ")
        except:
            print("0", end=" ")

    # print()
    # image = cv2.GaussianBlur(image, (3, 3), 3)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([38, 35, 75])
    upper_blue = np.array([92, 212, 255])

    # lower_blue = np.array([67, 64, 0])
    # upper_blue = np.array([97, 255, 255])

    # lower_blue = np.array([50, 37, 0])
    # upper_blue = np.array([97, 255, 255])

    # lower_blue = np.array([0, 0, 221])
    # upper_blue = np.array([180, 30, 255])

    threshold = cv2.inRange(hsv, lower_blue, upper_blue)
    threshold = cv2.bitwise_not(threshold)
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.erode(threshold, kernel, iterations=3)
    threshold = cv2.dilate(threshold, kernel, iterations=3)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('image', image)
    # cv2.imshow('canny',image_filtering2)
    cv2.imshow('threshold', threshold)
    # cv2.imshow('textImg', textImg)
    key = cv2.waitKey(1)

    # img_Sobel img_Laplacian img_canny image_filtering2
    if key == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

