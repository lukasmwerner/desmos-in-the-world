from dataclasses import dataclass
import math

import cv2
import numpy as np

from geometry import Box

import os
from google import genai
from dotenv import load_dotenv
from PIL import Image
import sympy

tags_dict = {}
for i in range(200, 400):
    tags_dict[i] = None


def make_gemini_client():
    client = genai.Client(api_key= os.getenv("GEMINI_API_KEY"))
    return client


@dataclass
class EquationComponent:
    id : int
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

        client = make_gemini_client()
        img = Image.fromarray(warped_image)
        try:
            eqn = sympy.sympify(client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                img,
                'ONLY PROVIDE THE SYMPY STRING REQUESTED, DO NOT INCLUDE QUATATIONS. THIS IS NOT A PYTHON SCRIPT DO NOT WRITE ANY PYTHON EVER. Given this image, generate a string of the equation provided in sympy format in order for sympy to underestand in the context of a function call for evaluating a mathematical. Example: (sin(x) - 2*cos(y)**2 + 3*tan(z)**3)**20)'
                ]
            ).text)
            tags_dict[self.id] = eqn
        except ValueError as e:
            return ""

        return eqn
