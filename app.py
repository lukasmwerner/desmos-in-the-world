import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

dict_key = cv2.aruco.DICT_4X4_1000
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)

source = cv2.VideoCapture(int(os.getenv("WEBCAM_ID")))
win_name = "desmos-irl"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cv2.waitKey(1) != ord('q'):
    has_frame, frame = source.read()
    blank_frame = np.zeros_like(frame)
    if not has_frame:
        break

    marker_corners, marker_ids, rejectedImgs = aruco_detector.detectMarkers(frame)
    detected_marker_img = cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
    cv2.imshow(win_name, detected_marker_img)

source.release()
cv2.destroyAllWindows()
