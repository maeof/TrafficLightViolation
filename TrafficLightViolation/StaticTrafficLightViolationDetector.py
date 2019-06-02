from TrafficLight import TrafficLightStatus
from TrafficLight import TrafficLight
from TrafficLightViolation import TrafficLightSignalDetectorStrategy
from TrafficLightViolation import VehiclePositionDetectorStrategy

import time
import calendar
import cv2 as cv

def isWithinBoundaries(rule, position):
    ox, oy = position
    rx, ry, rw, rh = rule

    if ox > rx and ox < rx + rw and oy > ry and oy < ry + rh:
        return True
    else:
        return False

def writeImage(frame, idx):
    ts = calendar.timegm(time.gmtime())
    cv.imwrite("output_images/" + "violation_{0}_{1}.png".format(ts, idx), frame)

class StaticTrafficLightViolationDetector:
    def __init__(self, trafficLight = TrafficLight([0, 0, 0, 0], [0, 0, 0, 0]),
                 trafficLightSignalDetectorStrategy = TrafficLightSignalDetectorStrategy.MorphologicalTrafficLightSignalDetectorStrategy(),
                 vehiclePositionDetectorStrategy = VehiclePositionDetectorStrategy.MorphologicalVehiclePositionDetectorStrategy()):
        self._trafficLight = trafficLight
        self._trafficLightSignalDetectorStrategy = trafficLightSignalDetectorStrategy
        self._vehiclePositionDetectorStrategy = vehiclePositionDetectorStrategy
        self._idx = 0

    def detectTrafficLightViolations(self, frame):
        trafficLightStatus = self._trafficLightSignalDetectorStrategy.getTrafficLightSignal(self._trafficLight, frame)

        if trafficLightStatus == TrafficLightStatus.Red:
            positions = self._vehiclePositionDetectorStrategy.getVehiclePositions(frame)

            for position in positions:
                if isWithinBoundaries(self._trafficLight.rule, position):
                    self._idx += 1
                    writeImage(frame, self._idx)