#!/usr/bin/python
#-*- coding:utf-8 -*-

'''
This example illustrates how to use Hough Transform to find lines

Usage:
    houghlines.py [<image_name>]
    image argument defaults to ./pic1.png
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np
import sys
import math

cap = cv2.VideoCapture("IMG_1926.MOV")

while(True):

    ret, frame = cap.read()

    src = frame
    dst = cv2.Canny(src, 50, 200) # aplica o detector de bordas de Canny à imagem src
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR) # Converte a imagem para BGR para permitir desenho colorido


    if True: # HoughLinesP
    
        lines = cv2.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
        print("Used Probabilistic Rough Transform")
        print("The probabilistic hough transform returns the end points of the detected lines")
        a,b,c = lines.shape

        sizes = []

        for i in range(a):

            def line_size(lines):
                size = math.sqrt((lines[i][0][2] - lines[i][0][0])**2 + (lines[i][0][3] - lines[i][0][1])**2)
                return size

            size = line_size(lines)

            sizes.append(size)

            # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
            cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3)

        linha1 = max(sizes)
        indx1 = sizes.index(max(sizes))
        sizes[indx1] = 0

        linha2 = max(sizes)
        indx2 = sizes.index(linha2)

        x_inicial = int(round((lines[indx1][0][0] + lines[indx2][0][0])/2))
        x_final = int(round((lines[indx1][0][2] + lines[indx2][0][2])/2))
        y_inicial = int(round((lines[indx1][0][1] + lines[indx2][0][1])/2))
        y_final = int(round((lines[indx1][0][3] + lines[indx2][0][3])/2))

        cv2.line(cdst, (x_inicial, y_inicial), (x_final, y_final), (0, 255, 0), 3)

    else:    # HoughLines
        # Esperemos nao cair neste caso
        lines = cv2.HoughLines(dst, 1, math.pi/180.0, 50, np.array([]), 0, 0)
        a,b,c = lines.shape
        for i in range(a):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0, y0 = a*rho, b*rho
            pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
            pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3)
        print("Used old vanilla Hough transform")
        print("Returned points will be radius and angles")

    cv2.imshow("detected lines", cdst)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
