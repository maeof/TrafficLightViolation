import TrafficLightViolation.StaticTrafficLightViolationDetector as tlv
from TrafficLightViolation import TrafficLightSignalDetectorStrategy
from TrafficLightViolation import VehiclePositionDetectorStrategy
from TrafficLight import TrafficLight

import cv2 as cv

if __name__ == '__main__' :
    cap = cv.VideoCapture("E:\\b\\video_data\\alberta1.mp4")

    MODEL_NAME = "E:\\b\\use_trainedmodel\\traffic-light-detector\\faster_rcnn_resnet101_coco_11_06_2017"
    PATH_TO_LABELS = "E:\\b\\use_trainedmodel\\traffic-light-detector\\mscoco_label_map.pbtxt"
    NUM_CLASSES = 90

    _, frame = cap.read()
    frame = cv.resize(frame, (640, 480))
    tfpos = cv.selectROI("out", frame)
    rulepos = cv.selectROI("out", frame)

    trafficLight = TrafficLight(tfpos, rulepos)
    detector = tlv.StaticTrafficLightViolationDetector(trafficLight)

    #detector = tlv.StaticTrafficLightViolationDetector(trafficLight,
    #                                                   TrafficLightSignalDetectorStrategy.MorphologicalTrafficLightSignalDetectorStrategy(),
    #                                                   VehiclePositionDetectorStrategy.TensorflowVehiclePositionDetectorStrategy(MODEL_NAME, PATH_TO_LABELS, NUM_CLASSES))

    frameidx = 0
    while (1):
        _, frame = cap.read()

        frameidx += 1
        frame = cv.resize(frame, (640, 480))
        detector.detectTrafficLightViolations(frame)

        cv.imshow("out", frame)

        k = cv.waitKey(1) & 0xff
        if k == 27:
            break

    cap.release()
