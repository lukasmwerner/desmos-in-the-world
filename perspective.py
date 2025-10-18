import cv2
import numpy as np

# Example points to transform
points_to_transform = np.array([[[100, 100], [150, 150]]], dtype=np.float32)

# Apply the transformation matrix M (obtained from getPerspectiveTransform)
transformed_points = cv2.perspectiveTransform(points_to_transform, M)

print(transformed_points)
