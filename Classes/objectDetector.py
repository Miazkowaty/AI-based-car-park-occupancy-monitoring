import cv2
import numpy


class objectDetector:

    def __init__(self, _haar_cascade_path, _scaleFactor, _minNeighbors, _minSize, _maxSize, _overlapThresh, _color, _thickness):

        self.haar_cascade = cv2.CascadeClassifier(_haar_cascade_path)
        self.scaleFactor = _scaleFactor
        self.minNeighbors = _minNeighbors
        self.minSize = _minSize
        self.maxSize = _maxSize
        self.overlapThresh = _overlapThresh
        self.color = _color
        self.thickness = _thickness

        return None


    def __non_max_suppression(self, boxes):

        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == 'i':
            boxes = boxes.astype('float')

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # computing the area of bounding boxes and sorting them
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = numpy.argsort(y2)

        # decreasing the number of overlaping boxes
        while len(idxs) > 0:

            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # finding the largest (x, y) coordinates for the start of
            # bounding box and the smallest (x, y) coordinates
            # for the end of bounding box
            xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
            yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
            xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
            yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

            # computing the width and height of bounding box
            w = numpy.maximum(0, xx2 - xx1 + 1)
            h = numpy.maximum(0, yy2 - yy1 + 1)

            # computing the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # deleting all of unnecessary indexes from list
            idxs = numpy.delete(idxs, numpy.concatenate(([last], numpy.where(overlap > self.overlapThresh)[0])))

        return boxes[pick].astype('int')


    def detect(self, image):

        objects = self.haar_cascade.detectMultiScale(image, scaleFactor = self.scaleFactor, minNeighbors = self.minNeighbors, minSize = self.minSize, maxSize = self.maxSize)
        objects = self.__non_max_suppression(objects)

        for (x, y, w, h) in objects:
            cv2.rectangle(image, (x, y), (x + w, y + h), self.color, self.thickness)

        return image, len(objects)