import cv2
import numpy as np

img = np.zeros((500, 500))
sws = np.zeros((200, 200))
dsd = np.zeros((660, 660))

ls = [img, sws, dsd]

for n,w in enumerate(ls):

    cv2.imshow(str(ls[n]),w)
    cv2.waitKey(5000)




