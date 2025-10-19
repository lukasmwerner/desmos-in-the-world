import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from screeninfo import get_monitors
import geometry
import components


def main():
    dict_key = cv2.aruco.DICT_4X4_1000
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)

    monitor = get_monitors()[0]

    source = cv2.VideoCapture(int(os.getenv("WEBCAM_ID")))
    win_name = "desmos-irl"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with open("homography/camera_to_monitor.pck", "rb") as pickle_file:
        camera_to_monitor = pickle.load(pickle_file)

    while cv2.waitKey(1) != ord("q"):
        has_frame, frame = source.read()
        blank_frame = np.zeros((monitor.height, monitor.width, 3), dtype=np.uint8)
        if not has_frame:
            break

        marker_corners, marker_ids, rejectedImgs = aruco_detector.detectMarkers(frame)

        if marker_ids is None:
            continue

        corners = dict(
            zip(
                (id_[0] for id_ in marker_ids), (corner[0] for corner in marker_corners)
            )
        )

        for id, corner in corners.items():
            if id % 4 == 0:
                possible_ids = list(range(id, id + 4))
                if all(map(lambda pid: pid in corners, possible_ids)):
                    box = geometry.Box(
                        *(geometry.Marker(*corners[pid]) for pid in possible_ids)
                    )
                    create_component(id, box, frame)

        # for box in boxes:
        #     inner_coordinates = cv2.perspectiveTransform(
        #         box.inner_coordinates().astype(np.float32).reshape(-1, 1, 2),
        #         camera_to_monitor,
        #     )

        #     cv2.polylines(
        #         blank_frame,
        #         inner_coordinates.astype(np.int32).reshape((1, -1, 1, 2)),
        #         True,
        #         (255, 255, 0),
        #         20,
        #     )

        cv2.imshow(win_name, blank_frame)

    source.release()
    cv2.destroyAllWindows()


def create_component(id, box, frame):
    category = id // 200
    if category == 0:
        # Graph
        pass
    elif category == 1:
        # Equations
        component = components.EquationComponent(box, frame)
        component.get_content()
    elif category == 2:
        # Add
        pass
    elif category == 3:
        # Range input
        pass
    else:
        # Gemini
        pass


if __name__ == "__main__":
    load_dotenv()
    main()
