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
    if img is None:
        print('Failed to load image file:', image_path)


    def onmouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print(x)
            print(y)

    display_fn('image', img)
    cv2.setMouseCallback('image', onmouse)

    cv2.waitKey(50000)


if __name__ == '__main__':
    process_board_picture('/Users/niddhogg/Documents/board_processing/BoardProcessing/board_processing/c7.jpg')