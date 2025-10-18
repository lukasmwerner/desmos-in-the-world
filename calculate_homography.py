import cv2

dict_key = cv2.aruco.DICT_4X4_1000
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)

win_name = "calibration_win"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



source.release()
cv2.destroyAllWindows()
