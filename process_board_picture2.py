"""
Author: Bogdan Kulynych (hello@bogdankulynych.me)
"""

import math
import numpy as np
import cv2

from aset import ApproximateSet


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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('pat1.jpg',0)
    w, h = template.shape[::-1]
    
    method = cv2.TM_CCOEFF
    res = cv2.matchTemplate(img,template,method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img,top_left, bottom_right, 255, 2)






    display_fn('lines_' + image_path, img)
    





    cv2.waitKey(50000)


if __name__ == '__main__':
    process_board_picture('t1.jpg')
