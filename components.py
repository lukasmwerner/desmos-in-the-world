from dataclasses import dataclass
import math

import cv2
import numpy as np

from geometry import Box


@dataclass
class EquationComponent:
    box: Box
    frame: np.ndarray

    # Warp box to a rectangle
    # pass to gemini.py
    # returns a sympy object
    def get_content(self):
        # Get source coordinates (inner corners of the box)
        src = self.box.inner_coordinates().astype(np.float32)

        # Calculate the width and height of the destination rectangle
        rectangle_width = int(np.linalg.norm(src[0] - src[1]))
        rectangle_height = int(np.linalg.norm(src[1] - src[2]))

        # Define destination points (a rectangle)
        dest = np.array(
            [
                [0, 0],
                [rectangle_width, 0],
                [rectangle_width, rectangle_height],
                [0, rectangle_height],
            ],
            dtype=np.float32,
        )

        # Get the perspective transform
        transform = cv2.getPerspectiveTransform(src, dest)

        # Warp the image
        warped_image = cv2.warpPerspective(
            self.frame, transform, (rectangle_width, rectangle_height)
        )

        cv2.imwrite("warped_eq.png", warped_image)
        return warped_image


@dataclass
class GraphComponent:
    id: int
    box: Box
    frame: np.ndarray

    # Render a graph on the picture
    # warped to the box.
    def render(self, canvas_bgr: np.ndarray, camera_to_monitor: np.ndarray):
        graph_bgr = cv2.imread("graph.png", cv2.IMREAD_COLOR)

        gh, gw = graph_bgr.shape[:2]
        source = np.array([[0, 0], [gw, 0], [gw, gh], [0, gh]], dtype=np.float32)

        # Map the box inner corners from camera -> monitor
        inner_cam = self.box.inner_coordinates().astype(np.float32).reshape(-1, 1, 2)
        inner_mon = (
            cv2.perspectiveTransform(inner_cam, camera_to_monitor)
            .reshape(-1, 2)
            .astype(np.float32)
        )

        # Homography: graph image -> monitor space
        H_graph_to_monitor = cv2.getPerspectiveTransform(source, inner_mon)

        out_h, out_w = canvas_bgr.shape[:2]
        warped = cv2.warpPerspective(
            graph_bgr,
            H_graph_to_monitor,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Simple binary mask: copy non-black pixels
        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_bool = mask.astype(bool)

        # Paste onto canvas
        canvas_bgr[mask_bool] = warped[mask_bool]
