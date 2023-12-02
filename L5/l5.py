# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:00:00 2019

@author: DELL
"""

import cv2
url = "D:\CS\DL_Tutorial\L5"
img = cv2.imread(url + "\gray.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('GRAY.JPG', img)
cv2.imshow('gray.jpg', gray)
cv2.waitKey(0)