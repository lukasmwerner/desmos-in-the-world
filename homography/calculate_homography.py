import cv2
from screeninfo import get_monitors
import numpy as np
from dotenv import load_dotenv
import os
import time
import pickle

load_dotenv("../.env")

def resize_image(inner_image, outer_image):
    w1, h1 = inner_image.shape[1], inner_image.shape[0]
    w2, h2 = outer_image.shape[1], outer_image.shape[0]

    scale = min(w2 / w1, h2 / h1)
    return cv2.resize(inner_image, (int(w1 * scale), int(h1 * scale)))

def main():
    monitor = get_monitors()[0]

    dict_key = cv2.aruco.DICT_4X4_1000
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)

    win_name = "calibration_win"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Project a checkerboard
    monitor_frame = np.zeros((monitor.height, monitor.width, 3), dtype=np.uint8)
    checkerboard = cv2.imread("pattern.png")
    resized_image = resize_image(checkerboard, monitor_frame)
    rs_h, rs_w = resized_image.shape[:2]

    monitor_frame[:rs_h, :rs_w] = resized_image
    ok, monitor_corners = cv2.findChessboardCorners(image, (9, 6))

    cv2.imshow(win_name, monitor_frame)

    time.sleep(250)
    source = cv2.VideoCapture(os.getenv("WEBCAM_ID"))
    has_frame, frame = source.read()

    ok, camera_corners = cv2.findChessboardCorners(frame, (9, 6))

    camera_to_monitor = cv2.findHomography(camera_corners, monitor_corners)
    with open("camera_to_monitor.pck", "wb") as pickle_file:
        pickle.dump(camera_to_monitor, pickle_file)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
