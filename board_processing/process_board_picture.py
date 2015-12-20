"""
Author:
Vladimir Duchenchuk
Bogdan Kulynych (hello@bogdankulynych.me)
"""

import math
import numpy as np
import cv2
from docopt import docopt

from aset import ApproximateSet

__usage = """
Process Board Image

Usage:
  process_board_picture [<image>]
"""

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

    # Load reference image
    src = cv2.imread(image_path)
    img = src
    if img is None:
        print('Failed to load image file:', image_path)

    # Load current state image
    # img2 = cv2.imread("/Users/niddhogg/Documents/board_processing/BoardProcessing/board_processing/test_images/IMG_3485.jpg")
    # img2 = cv2.imread("test_images/test_1.jpg")



    # array of inputed coordinates
    pts = []

    # comment in xcode: command + /

#    def onmouse(event, x, y, flags, param):
#        if event == cv2.EVENT_LBUTTONUP:
#            if (len(pts) < 4):
#                cv2.circle(img,(x,y),3,(255,0,0),-1)
#                pts.append((x,y))
#
#                if len(pts) == 2:
#                    cv2.line(img,pts[0],pts[1],(0,255,0),2)
#                elif len(pts) == 3:
#                    cv2.line(img,pts[1],pts[2],(0,255,0),2)
#                elif len(pts) == 4:
#                    cv2.line(img,pts[2],pts[3],(0,255,0),2)
#                    cv2.line(img,pts[3],pts[0],(0,255,0),2)
#
#
#
#
#    #display_fn('image', img)
#    cv2.namedWindow('image')
#    cv2.setMouseCallback('image', onmouse)
#
#    while(1):
#        cv2.imshow('image',img)
#        if cv2.waitKey(20) & 0xFF == 27:
#            break
#
#    cv2.destroyAllWindows()

    # test only
#    pts = [(116, 334), (346, 334), (385, 577), (59, 577)]
    pts = [(108, 30), (418, 32), (427, 349), (91, 346)]

    # do transform
    pts2 = np.array(pts, dtype="float32")
    original = cv2.imread(image_path)

    # get warped image
    warped = four_point_transform(original, pts2)

    # resize warped image
    size = 400
    warped_resized = cv2.resize(warped, (size, size))



    print(pts)

    # processing part

    warped_resized_copy = warped_resized.copy()
    # now let's draw the lines for debug
    for i in range(1,8):
        num = i * (size/8)
        cv2.line(warped_resized_copy,(0,num),(size,num),(0,255,0),1)
        cv2.line(warped_resized_copy,(num,0),(num,size),(0,255,0),1)

    cv2.imshow("Original",original )
    cv2.imshow("Warped ", warped_resized_copy)


    img_board = []

    for i in range(0,8):
        for j in range(0,8):
            start_x = i*(size/8)
            start_y = j*(size/8)

            end_x = (i+1)*(size/8)
            end_y = (j+1)*(size/8)

            sqr_img = warped_resized[start_x:end_x, start_y:end_y].copy()
            img_board.append( (sqr_img,i,j) )

    # ok, got 64 images, now let's process them

    # here we form up matrix
    Matrix = [[0 for x in range(8)] for x in range(8)]

    for (sqr_img,img_i,img_j) in img_board:
        size_sqr = size / 8
        start = size_sqr / 3
        end = size_sqr - start
        sqr_img_center = sqr_img[start:end, start:end].copy()
        
        # output
        # print(sum(sum(sqr_img_center))[2])
        
        sum_r = 0
        sum_b = 0
        
        for i in range(start, end):
            for j in range(start, end):
                pixel_value = sqr_img[i][j]
                b = pixel_value[0]
                g = pixel_value[1]
                r = pixel_value[2]
                
                if (r > 220) & ( b < 50) & ( g < 50):
                    sum_r = sum_r + r
                
                if (b > 220) & ( r < 50):
                    sum_b = sum_b + b


        if (sum_r > 10000):
            Matrix[img_i][img_j] = 1
            # cv2.imshow("Red", sqr_img_center )
            # cv2.waitKey(0)


        if (sum_b > 10000):
            Matrix[img_i][img_j] = 2
            # cv2.imshow("Blue",sqr_img_center )
            # cv2.waitKey(0)


    # print(Matrix)
    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
        for row in Matrix]))


    cv2.waitKey(0)

if __name__ == '__main__':
    # process_board_picture('/Users/niddhogg/Documents/board_processing/BoardProcessing/board_processing/test_images/IMG_3472.jpg')
    arguments = docopt(__usage, version='Board Processing 0.0.1')

    image = arguments['<image>']

    if image is None:
        image = 'test_images/test_1.jpg'

    process_board_picture(image)
