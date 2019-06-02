import abc
from TrafficLight import TrafficLight
from TrafficLight import TrafficLightStatus

import cv2 as cv
import numpy as np

class TrafficLightSignalDetectorStrategyAbstract(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getTrafficLightSignal(self, trafficLight, frame):
        """Required Method"""

class MorphologicalTrafficLightSignalDetectorStrategy(TrafficLightSignalDetectorStrategyAbstract):
    def getTrafficLightSignal(self, trafficLight, frame):
        x, y, w, h = trafficLight.position
        crop_frame = frame[y:y + h, x:x + w]

        if self._isForbiddenSignal(crop_frame, w, h):
            trafficLight.status = TrafficLightStatus.Red
        else:
            trafficLight.status = TrafficLightStatus.Green

        return trafficLight.status

    def _isForbiddenSignal(self, img, w, h, Threshold=0.05):
        img_y = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

        minYCB = np.array([0, 150, 0])
        maxYCB = np.array([255, 255, 140])

        maskYCB = cv.inRange(img_y, minYCB, maxYCB)
        rate = np.count_nonzero(maskYCB) / (w * h)

        if rate > Threshold:
            return True
        else:
            return False