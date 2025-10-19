import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import typing
from dotenv import load_dotenv
from dataclasses import dataclass


@dataclass
class Marker:
    # this type is kinda scuffed? but this is a hackathon
    tl: typing.Iterable[int]
    tr: typing.Iterable[int]
    br: typing.Iterable[int]
    bl: typing.Iterable[int]


@dataclass
class Box:
    tl: Marker
    tr: Marker
    bl: Marker 
    br: Marker

    def inner_coordinates(self):
        return np.array((
            self.tl.br,
            self.tr.bl,
            self.br.tl,
            self.bl.tr,
        ), np.int32)

    def outer_coordinates(self):
        return np.array((
            self.tl.tl,
            self.tr.tr,
            self.br.br,
            self.bl.bl,
        ), np.int32)



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
    # detected_marker_img = cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    print(marker_corners)
    corners = dict(zip((id_[0] for id_ in marker_ids), (corner[0] for corner in marker_corners)))

    boxes = []
    for id, corner in corners.items():
        if id % 4 == 0:
            possible_ids = list(range(id, id+4))
            if all(map(lambda pid: pid in corners, possible_ids)):
                boxes.append(Box(*(Marker(*corners[pid]) for pid in possible_ids)))

    for box in boxes:
        cv2.polylines(
            frame,
            box.inner_coordinates().reshape((1, -1, 1, 2)),
            True,
            (255, 255, 0),
            20,
        )

source.release()
cv2.destroyAllWindows()
