import numpy as np
import cv2

# load the image and convert it to grayscale
image = cv2.imread("zp.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
_, cnts, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imwrite('Image.jpg', image)
cv2.imshow("Image", image)
cv2.waitKey(0)
