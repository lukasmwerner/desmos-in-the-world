from collections import defaultdict
from dataclasses import dataclass, field
import math

from geometry import Box

import cv2
import numpy as np
import os
import sympy
import typing
from google import genai
from dotenv import load_dotenv
from PIL import Image

tags_dict = {}
for i in range(200, 400):
    tags_dict[i] = None


def make_gemini_client():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client


@dataclass
class EquationComponent:
    id: int
    box: Box
    frame: np.ndarray
    does_output = True

    # Warp box to a rectangle
    # pass to gemini.py
    # returns a sympy object
    def compute_content(self):
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

        client = make_gemini_client()
        img = Image.fromarray(warped_image)
        try:
            if tags_dict[self.id] != None:
                raise ValueError()
            eqn = sympy.sympify(
                client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        img,
                        "ONLY PROVIDE THE SYMPY STRING REQUESTED, DO NOT INCLUDE QUATATIONS. THIS IS NOT A PYTHON SCRIPT DO NOT WRITE ANY PYTHON EVER. Given this image, generate a string of the equation provided in sympy format in order for sympy to underestand in the context of a function call for evaluating a mathematical. Example: (sin(x) - 2*cos(y)**2 + 3*tan(z)**3)**20)",
                    ],
                ).text
            )
            tags_dict[self.id] = eqn
        except ValueError as e:
            return ""

        return eqn


@dataclass
class GraphComponent:
    id: int
    box: Box
    frame: np.ndarray
    inputs: list
    equation: sympy.Expr
    graph: dict = field(default_factory=dict)
    does_output = False

    DISPLAY_DENSITY = 1

    def eqn_to_bytearray(self) -> None:
        p = sympy.plotting.plot((self.equation, (-5, 5)), show=False)
        p.process_series()

        canvas = p.fig.canvas
        canvas.draw()
        w, h = canvas.get_width_height()
        self.graph = np.frombuffer(
            canvas.buffer_rgba().tobytes(), dtype=np.uint8
        ).reshape((w * self.DISPLAY_DENSITY, h * self.DISPLAY_DENSITY, 4))[:, :, 1:]
        p.save('graph.png')
        p.close()


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


@dataclass
class GeminiComponent:
    expired: bool = False

    def expire(self):
        expired = True
