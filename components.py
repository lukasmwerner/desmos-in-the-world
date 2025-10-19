from dataclasses import dataclass
import math

import cv2
import numpy as np

from geometry import Box


@dataclass
class EquationComponent:
    box: Box

    # Warp box to a rectangle
    # pass to gemini.py
    # returns a sympy object
    def get_content(self):
        rectangle_width = np.linalg.norm(self.box.tl.br - self.box.tr.bl)
        rectangle_height = np.linalg.norm(self.box.tr.bl - self.box.br.tl)

        dest = np.array(
            [
                [0, 0],
                [rectangle_width, 0],
                [rectangle_width, rectangle_height],
                [0, rectangle_height],
            ]
        )

        transform = cv2.getPerspectiveTransform(self.box.inner_coordinates, dest)
        warped_image = cv2.warpPerspective(
            self.box, transform, (rectangle_width, rectangle_height), dest
        )

        cv2.imwrite("warped_eq.png", warped_image)
