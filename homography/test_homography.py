import pickle
import cv2
import os
from dotenv import load_dotenv
from screeninfo import get_monitors
import numpy as np

load_dotenv("../.env")

with open("camera_to_monitor.pck", "rb") as pickle_file:
    camera_to_monitor = pickle.load(pickle_file)

dict_key = cv2.aruco.DICT_4X4_1000
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)

monitor = get_monitors()[0]
monitor_frame = np.zeros((monitor.height, monitor.width, 3), dtype=np.uint8)

source = cv2.VideoCapture(int(os.getenv("WEBCAM_ID")))

win_name = "calibration_test_win"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cv2.waitKey(1) != ord("q"):
    monitor_frame = np.zeros((monitor.height, monitor.width, 3), dtype=np.uint8)

    has_frame, frame = source.read()
    if not has_frame:
        break

    marker_corners, marker_ids, rejectedImgs = aruco_detector.detectMarkers(frame)

    for marker_corner_list in marker_corners:
        points = cv2.perspectiveTransform(
            marker_corner_list.reshape(-1, 1, 2), camera_to_monitor
        )
        monitor_frame = cv2.fillPoly(
            monitor_frame, [points.astype(np.uint8)], (0, 255, 255)
        )

    cv2.imshow(win_name, monitor_frame)

source.release()
cv2.destroyAllWindows()
