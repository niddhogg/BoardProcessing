"""
Author: Bogdan Kulynych (hello@bogdankulynych.me)
"""

import math
import numpy as np
import cv2

from aset import ApproximateSet


# http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
        
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
        
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
   
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
        
 
    dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")
                        
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
   
    return warped



def process_board_picture(image_path, display_mode='show'):
    # Pick display mode
    assert display_mode in ['show', 'write']
    if display_mode == 'show':
        display_fn = cv2.imshow
    elif display_mode == 'write':
        display_fn = cv2.imwrite

    # Load image
    src = cv2.imread(image_path)
    img = src
    if img is None:
        print('Failed to load image file:', image_path)

    # array of inputed coordinates
    pts = []

    def onmouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if (len(pts) < 4):
                cv2.circle(img,(x,y),8,(255,0,0),-1)
                pts.append((x,y))

                if len(pts) == 2:
                    cv2.line(img,pts[0],pts[1],(0,255,0),4)
                elif len(pts) == 3:
                    cv2.line(img,pts[1],pts[2],(0,255,0),4)
                elif len(pts) == 4:
                    cv2.line(img,pts[2],pts[3],(0,255,0),4)
                    cv2.line(img,pts[3],pts[0],(0,255,0),4)




    #display_fn('image', img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onmouse)

    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    print(pts)

# pts2 = np.array(eval("[(73, 239), (356, 117), (475, 265), (187, 443)]"), dtype = "float32")
    pts2 =  [list( e) for e in pts]

    print(pts2)

    warped = four_point_transform(img, pts2)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)



    # cv2.waitKey(50000)


if __name__ == '__main__':
    process_board_picture('/Users/niddhogg/Documents/board_processing/BoardProcessing/board_processing/c7.jpg')