"""
Author: Bogdan Kulynych (hello@bogdankulynych.me)
"""

import math
import numpy as np
import cv2

from aset import ApproximateSet


def process_board_picture(image_path, display_mode='show'):
    # Parameters
    CANNY_THRESHOLD = 50                   # %
    NORMAL_WIDTH = 640                     # pixels
    HOUGH_DIST_THRESHOLD = 1               # pixels
    HOUGH_ANGLE_THRESHOLD = math.pi / 180  # radians
    HOUGH_VOTES_THRESHOLD = 95             # number of votes
    LINE_SIMILARITY_THRESHOLD = 2 * 0.25

    # Pick display mode
    assert display_mode in ['show', 'write']
    if display_mode == 'show':
        display_fn = cv2.imshow
    elif display_mode == 'write':
        display_fn = cv2.imwrite

    # Load image
    src = cv2.imread(image_path)
    img = src

    # Calculate parameters
    width, height = img.shape[0], img.shape[1]
    ratio = min(float(NORMAL_WIDTH) / width, 1)

    # Filters
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
    img = cv2.equalizeHist(img)
    img = cv2.blur(img, (3, 3))
    img = cv2.Canny(img, CANNY_THRESHOLD, CANNY_THRESHOLD*3)

    # Contours
    contour_image = np.array(img)
    contours, _  = cv2.findContours(contour_image, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = cv2.convexHull(max(contours, key=cv2.contourArea))
    cv2.drawContours(contour_image, [largest_contour], -1, (255, 255, 255), 2)

#display_fn('contour_' + image_path, contour_image)
# cv2.waitKey(10000)

    # Global line detector (hough)
    raw_lines = cv2.HoughLines(img, HOUGH_DIST_THRESHOLD,
                               HOUGH_ANGLE_THRESHOLD,
                               HOUGH_VOTES_THRESHOLD)

    # All detected lines are put into approximate set data structure
    # with following similarity metric:
    def line_metric(x, y):
        """
        Calculates metric value for two 2D points x and y:
        = sqrt( eta[0] * (x[0] - y[0])**2 + eta[1] * (x[1] - y[1])**2 )

        where eta is normalizing vector
        """
        width, height = img.shape[0], img.shape[1]
        normalizer = np.array([1. / math.sqrt(width**2 + height**2),
                               1. / (0.5 * math.pi)])
        diffs = np.sqrt(normalizer) * (x - y)
        return np.sqrt(np.dot(diffs, diffs))

    # Init approximate set
    lines = ApproximateSet(LINE_SIMILARITY_THRESHOLD, metric=line_metric)
    for line in raw_lines[0]:
        lines.add(line)


    # Draw detected global lines
    lines_image = np.array(img)

    for centroid in lines:
        rho, theta = centroid[0], centroid[1]
        x0 = rho * math.cos(theta)
        y0 = rho * math.sin(theta)
        width, height = img.shape
        pt1 = int(x0 + 5*(-math.sin(theta))), int(y0 + 5*(math.cos(theta)))
        pt2 = int(x0 - 5*(-math.sin(theta))), int(y0 - 5*(math.cos(theta)))
        cv2.line(lines_image, pt1, pt2, (255, 255, 255), 2)

    display_fn('lines_' + image_path, lines_image)
    cv2.waitKey(10000)


if __name__ == '__main__':
    process_board_picture('c7.jpg')
