import cv2
from matplotlib import pyplot as plt 
import numpy as np

image = cv2.imread('one rupee 1.jpg')
#image = cv2.imread()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create()
keypoints=fast.detect(gray, None)

kpimg = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("FAST", kpimg)
print("Keypoints = " + str(len(keypoints)))
cv2.waitKey(0)
cv2.destroyAllWindows()
