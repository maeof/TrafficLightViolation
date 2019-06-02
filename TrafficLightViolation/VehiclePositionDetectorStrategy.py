import abc
import cv2 as cv
import numpy as np
from TrafficLightViolation import TensorFlowObjectDetector

kernel33 = np.ones((3, 3), np.uint8)

class VehiclePositionDetectorStrategyAbstract(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getVehiclePositions(self, frame):
        """Required Method"""

class MorphologicalVehiclePositionDetectorStrategy(VehiclePositionDetectorStrategyAbstract):
    def __init__(self):
        self._fgbg = cv.createBackgroundSubtractorMOG2(history=4000)
        self._contours = []

    def getVehiclePositions(self, frame):
        fgmask = self._fgbg.apply(frame)

        ret, thresh = cv.threshold(fgmask, 254, 255, cv.THRESH_BINARY)
        erode = cv.morphologyEx(thresh, cv.MORPH_ERODE, kernel33, iterations=1)
        opening = cv.morphologyEx(erode, cv.MORPH_OPEN, kernel33, iterations=1)
        dilate = cv.dilate(opening, kernel33, iterations=1)
        closing = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel33, iterations=6)

        contours, _ = cv.findContours(np.uint8(closing), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self._contours = self._getContoursOfInterest(contours)

        positions = [[-1, -1]]
        for contour in self._contours:
            massCenter = self._getMassCenter(contour)
            positions.insert(len(positions), [massCenter[0], massCenter[1]])

        return positions

    def _getContoursOfInterest(self, contours, minContourArea=2000):
        contoursOfInterest = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if max(w, h) > 2.25 * min(w, h):
                continue

            if cv.contourArea(contour) < minContourArea:
                continue

            contoursOfInterest.append(contour)
        return contoursOfInterest

    def _getMassCenter(self, contour):
        moments = cv.moments(contour)
        return int(moments['m10'] / (moments['m00'] + 1e-5)), int(moments['m01'] / (moments['m00'] + 1e-5))

class TensorflowVehiclePositionDetectorStrategy(VehiclePositionDetectorStrategyAbstract):
    def __init__(self, model_name, path_to_labels, num_classes):
        self.tf = TensorFlowObjectDetector.TensorFlowObjectDetector(model_name, path_to_labels, num_classes)

    def getVehiclePositions(self, frame):
        (boxes, scores, classes, num) = self.tf.detectTF(frame)
        (boxes, scores, classes, num) = (np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32), np.squeeze(num))

        positions = [[-1, -1]]
        for i in range(0, boxes.shape[0]):
            if (classes[i] == 3 or classes[i] == 6 or classes[i] == 8) and (scores[i] > .8):
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                im_width = 640
                im_height = 480
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
                left, right, top, bottom = (int(left), int(right), int(top), int(bottom))
                rectW = right - left
                rectH = bottom - top
                rectCenterX = int(rectW / 2) + left
                rectCenterY = int(rectH / 2) + top
                positions.insert(len(positions), [rectCenterX, rectCenterY])

        return positions