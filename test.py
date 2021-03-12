import numpy as np
from ImplementMethod import ImplementDetectMethod
from ImageMethod import ImageDetectMethod
import cv2
import math
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)



cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)
# cap.set(cv2.CAP_PROP_SETTINGS, 1)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_FOCUS, 10)


while True:
    mode = 0
    ret, image = cap.read()
    image = cv2.flip(image,-1)

    if not ret:
        continue
    image = image[300:900, 500:1800]
    TextResult = np.zeros((40,40))
    TextImg_Canny = np.zeros_like(image)
    
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    radius = 5
    n = 1
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv2.dft(np.float32(image1), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)
 
    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)
 
    # 构建ButterWorth高通滤波掩模
 
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
            # 计算(i, j)到中心点的距离
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            try:
                mask[i, j, 0] = mask[i, j, 1] = 1 / (1 + pow(radius / d, 2*n))
            except ZeroDivisionError:
                mask[i, j, 0] = mask[i, j, 1] = 0
    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * mask
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv2.normalize(image_filtering, image_filtering, 0, 1, cv2.NORM_MINMAX)
    image_filtering2 = image_filtering * 255
    image_filtering2 = image_filtering2.astype('uint8')
    _, image_filtering3 = cv2.threshold(image_filtering2, 25, 255, cv2.THRESH_BINARY)

    img1 = image.copy()

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)

    img1 = cv2.filter2D(img1, -1, kernel=kernel)

    w = [0,0]
    if ret:
        
        g = ImageDetectMethod(img1)

        threshold_ = np.zeros((600,1300))

        cv2.line(image,(0,485),(1000,485),(255,255,255))
        threshold = cv2.cvtColor(g.Blur, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(threshold, 128, 255, cv2.THRESH_BINARY)
        
        circles = cv2.HoughCircles(image_filtering2, cv2.HOUGH_GRADIENT, 1, 180,
                                   param1=150, param2=20, minRadius=5, maxRadius=15)

        if circles is None:
            pass
        else:
            for i in circles[0, :]:
                cv2.circle(image, (int(i[0]), int(i[1])), int(int(i[2])), (0, 255, 0), 2)
                cv2.circle(image, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

                w[0] = int(i[0])
                w[1] = int(i[1])
        try:
            if circles is not None:
                textImg = image[w[1] + 20:w[1] + 60, w[0] - 20: w[0] + 20]
                TextImg = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)
                TextImg_Canny = cv2.Canny(TextImg, 50, 150, L2gradient=True)
                _, TextResult = cv2.threshold(TextImg_Canny, 85, 255, cv2.THRESH_BINARY)

        except:
            continue
    if np.all(circles==0) or circles is None:
        pass
    else:
        print(circles.shape)

    cv2.imshow('image',image)
    cv2.imshow('canny',image_filtering2)
    cv2.imshow('threshold_',image_filtering3)
    cv2.imshow('textImg', TextImg_Canny)
    key = cv2.waitKey(1)

    if key == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
