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
