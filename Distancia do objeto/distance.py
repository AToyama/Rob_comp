import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img = cv2.imread("luigi2.png",0)        

video = cv2.VideoCapture("video_luigi.mp4") 

while True:

    sift = cv2.SIFT()

    ret, img2 = video.read()

    kp1, des1 = sift.detectAndCompute(img,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        dst = cv2.perspectiveTransform(pts,M)

        img2b = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)        
        
        y1 = abs((dst[2][0][1] + dst[1][0][1]))
        y2 = abs((dst[3][0][1] + dst[0][0][1]))

        #altura
        delta_y = y1 - y2
        altura_cm = 12
        #altura em pixels inicial = 496
        #Distancia inicial = 30cm

        foco = 1240

        distancia = (altura_cm*foco)/delta_y
        #distancia = altura_pxl*(altura_cm/delta_y)
        print(distancia)




    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    cv2.imshow('frame',img2)
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()