"""

RPA: Region proposal algorithms

The algorithms will be used as primary step of the rcnn model

Region proposal algorithms identify prospective objects in an image using
segmentation. In segmentation, we group adjacent regions which are similar
to each other based on some criteria such as color, texture etc.
Unlike the sliding window approach where we are looking for the object at
all pixel locations and at all scales, region proposal algorithm work by
grouping pixels into a smaller number of segments. So the final number of
proposals generated are many times less than sliding window approach.
This reduces the number of image patches we have to classify.
These generated region proposals are of different scales and aspect ratios.

"""

import os
import sys

import cv2
import numpy as np


class Configs:
    def __init__(
        self,
        data_dir,
        num_classes,
        dropout_rate,
        learning_rate,
        test_size,
        image_shape,
        batch_size,
        epochs,
        optimizer,
        loss,
        metrics,
    ):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = (image_shape[0], image_shape[1], 3)
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics


def drawRectangle(image, rectangle):
    x, y, w, h = rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)


def showRectangles(image, rectangles, first_n):
    for i, rect in enumerate(rectangles):
        # draw reactangles
        drawRectangle(image, rect)
        if i == first_n:
            break
    # Show image
    cv2.imshow("Output", image)


def getROI(image, rect):
    x1, y1, x2, y2 = rect
    roi = image[y1:y2, x1:x2]
    return roi


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def selectiveSearch(method, image):
    # segmentation
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set the image on which we will run segmentation
    ss.setBaseImage(image)
    # Switch to fast but low recall selective search method
    if method == "fast":
        ss.switchToSelectiveSearchFast()
    elif method == "quality":
        ss.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)
    # process the selective search segmentatation on the image
    rectangles = ss.process()

    return rectangles


# Malisiewicz et al.
def non_max_suppression_fast(boxes, probs, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by probablities
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


if __name__ == "__main__":
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # Get the sample image
    path = os.getcwd()
    path = os.path.join(path, "vision/examples/2007_000027.jpg")
    print(path)
    image = cv2.imread(path)
    # Set the selective search object class
    rectangles = selectiveSearch("fast", image)
    # # Show rectangles
    showRectangles(image, rectangles, first_n=100)

    # record key press
    k = cv2.waitKey(0) & 0xFF
    # Destroy all the cv2 windows after pressing Esc
    if k == 27:  # Esc key to stop
        cv2.destroyAllWindows()

    print("Done")
