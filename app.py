from collections import defaultdict, deque
import pickle
import cv2
import numpy as np
import sympy.abc
import os
from dotenv import load_dotenv
from screeninfo import get_monitors
import geometry
import components
from shapely.geometry import Point, Polygon, LineString


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

        components = {}

        for id, corner in corners.items():
            if id % 4 == 0:
                possible_ids = list(range(id, id + 4))
                if all(map(lambda pid: pid in corners, possible_ids)):
                    box = geometry.Box(
                        *(geometry.Marker(*corners[pid]) for pid in possible_ids)
                    )
                    components[id] = create_component(id, box, frame)

        connections = connect_components(components, blank_frame, camera_to_monitor)
        process_components(components, blank_frame, camera_to_monitor, connections)
        cv2.imshow(win_name, blank_frame)

    source.release()
    cv2.destroyAllWindows()


def create_component(id, box, frame):
    category = id // 200
    if category == 0:
        # Graph
        component = components.GraphComponent(id, box, frame, [], sympy.abc.x**2, None)
        component.eqn_to_bytearray()
        return component
    elif category == 1:
        # Equations
        return components.EquationComponent(id, box, frame)
    elif category == 2:
        # Add
        pass
    elif category == 3:
        # Range input
        pass
    else:
        # Gemini
        pass


def process_components(components, blank_frame, camera_to_monitor, connections):
    indegrees = defaultdict(int)
    for component_id in connections:
        indegrees[connections[component_id]] += 1

    q = deque()

    for component_id in components:
        if indegrees[component_id] == 0:
            q.append(component_id)

    while q:
        component_id = q.popleft()

        # process component
        if hasattr(components[component_id], "get_content"):
            content = components[component_id].get_content()
            if component_id in connections:
                if hasattr(components[connections[component_id]], "inputs"):
                    components[connections[component_id]].inputs.append(content)

        if hasattr(components[component_id], "render"):
            components[component_id].render(blank_frame, camera_to_monitor)

        if component_id in connections:
            indegrees[connections[component_id]] -= 1
            if indegrees[connections[component_id]] == 0:
                q.append(connections[component_id])


def connect_components(components, blank_frame, camera_to_monitor):
    connections = {}

    for component_id in components:

        if not hasattr(component, "does_output") or component.does_output == False:
            continue

        component = components[component_id]
        outer_points = component.box.outer_coordinates()
        # transform outer points from camera to monitor plane using homography
        pts = outer_points.astype(np.float32).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts, camera_to_monitor).reshape(
            -1, 2
        )
        top_left, top_right, _, _ = transformed_pts

        for other_component_id in components:
            other_component = components[other_component_id]
            if component_id == other_component_id:
                continue

            other_outer_points = other_component.box.outer_coordinates()
            pts2 = other_outer_points.astype(np.float32).reshape(-1, 1, 2)
            transformed_other = cv2.perspectiveTransform(
                pts2, camera_to_monitor
            ).reshape(-1, 2)
            other_outer_points = transformed_other

            midpoint = (top_left + top_right) / 2.0

            vector_top = top_left - top_right
            vector_scale = 3.0
            orthogonal_vector = np.array(
                [-vector_top[1] * vector_scale, vector_top[0] * vector_scale],
                dtype=np.float32,
            )

            # Build shapely geometries from transformed monitor coordinates
            line = LineString([tuple(midpoint), tuple((midpoint + orthogonal_vector))])
            other_shape = Polygon(other_outer_points.tolist())

            intersection = line.intersection(other_shape)

            if not intersection.is_empty:
                connections[component_id] = other_component_id

            # extract a point to draw to (if available)
            found = None
            if intersection.is_empty:
                found = None
            elif intersection.geom_type == "Point":
                found = (intersection.x, intersection.y)
            elif intersection.geom_type in ("LineString", "LinearRing"):
                found = tuple(intersection.coords)[0]
            else:
                # try to get first geometry
                try:
                    geoms = list(intersection.geoms)
                    for g in geoms:
                        if g.geom_type == "Point":
                            found = (g.x, g.y)
                            break
                        if g.geom_type in ("LineString", "LinearRing"):
                            found = tuple(g.coords)[0]
                            break
                except Exception:
                    found = None

            if found is None:
                line_end = (midpoint + orthogonal_vector).astype(np.float32)
            else:
                line_end = np.array(found, dtype=np.float32)

            # draw using integer pixel coordinates
            p1 = tuple(np.round(midpoint).astype(int))
            p2 = tuple(np.round(line_end).astype(int))
            cv2.line(blank_frame, p1, p2, (0, 255, 255), 5)

    return connections


if __name__ == "__main__":
    load_dotenv()
    main()
