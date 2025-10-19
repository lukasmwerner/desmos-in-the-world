import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

source = cv2.VideoCapture(int(os.getenv("WEBCAM_ID")))
win_name = "desmos-irl"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cv2.waitKey(1) != ord("q"):
    has_frame, frame = source.read()
    if not has_frame:
        break

    cv2.imshow(win_name, frame)

source.release()
cv2.destroyAllWindows()
