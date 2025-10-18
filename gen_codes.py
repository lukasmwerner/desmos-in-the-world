import cv2
import os

dict_key = cv2.aruco.DICT_4X4_1000
dict_size = 1000

aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)

marker_dir = "markers"
os.mkdir(marker_dir)

for id in range(dict_size):
    output_dir = os.path.join(marker_dir, f"marker-{id:03}.png")
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, 256)
    cv2.imwrite(output_dir, marker_img)
