import asyncio
from collections import defaultdict, deque
import pickle
import cv2
import numpy as np
import sympy.abc
import os
from dotenv import load_dotenv
from screeninfo import get_monitors
from components import EquationComponent
import geometry
from components import *
from shapely.geometry import Point, Polygon, LineString


async def main():
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

    components = {}

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

        new_ids = set()

        for id, corner in corners.items():
            if id % 4 == 0:
                possible_ids = list(range(id, id + 4))
                if all(map(lambda pid: pid in corners, possible_ids)):
                    box = geometry.Box(
                        *(geometry.Marker(*corners[pid]) for pid in possible_ids)
                    )
                    if id in components:
                        components[id].box = box
                    else:
                        components[id] = create_component(id, box, frame)

                    new_ids.add(id)

        for key in list(components.keys()):
            if key not in new_ids:
                del components[key]

        connections = connect_components(components, blank_frame, camera_to_monitor)
        await process_components(
            components, blank_frame, camera_to_monitor, connections
        )

        # Draw border
        cv2.rectangle(
            blank_frame, (0, 0), (monitor.width, monitor.height), (255, 255, 255), 4
        )
        cv2.imshow(win_name, blank_frame)

    source.release()
    cv2.destroyAllWindows()


def create_component(id, box, frame):
    category = id // 200
    if category == 0:
        # Graph
        component = GraphComponent(id, box, frame, set())
        component.eqn_to_bytearray()
        return component
    elif category == 1:
        # Equations
        return EquationComponent(id, box, frame)
    elif category == 2:
        # Add
        pass
    else:
        # Range input
        pass


async def process_components(components, blank_frame, camera_to_monitor, connections):
    indegrees = defaultdict(int)
    for component_id in connections:
        indegrees[connections[component_id]] += 1

    q = deque()

    for component_id in components:
        if isinstance(components[component_id], EquationComponent):
            task = asyncio.create_task(components[component_id].compute_content())
        elif isinstance(components[component_id], GraphComponent):
            components[component_id].inputs.clear()
        if indegrees[component_id] == 0:
            q.append(component_id)

    while q:
        await asyncio.sleep(0)
        component_id = q.popleft()

        if (
            isinstance(components[component_id], EquationComponent)
            and (component_id in connections)
            and isinstance(components[connections[component_id]], GraphComponent)
        ):
            components[connections[component_id]].inputs.add(
                tags_dict[components[component_id].id]
            )

        if hasattr(components[component_id], "render"):
            components[component_id].render(blank_frame, camera_to_monitor)

        if component_id in connections:
            target = components[connections[component_id]]

            indegrees[connections[component_id]] -= 1
            if indegrees[connections[component_id]] == 0:
                q.append(connections[component_id])


def connect_components(components, blank_frame, camera_to_monitor):
    connections = {}

    for component_id in components:

        component = components[component_id]
        if not hasattr(component, "does_output") or component.does_output == False:
            continue

        outer_points = component.box.outer_coordinates()
        # transform outer points from camera to monitor plane using homography
        pts = outer_points.astype(np.float32).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts, camera_to_monitor).reshape(
            -1, 2
        )
        top_left, top_right, _, _ = transformed_pts

        midpoint = (top_left + top_right) / 2.0

        vector_top = top_left - top_right
        vector_scale = 3.0
        orthogonal_vector = np.array(
            [-vector_top[1] * vector_scale, vector_top[0] * vector_scale],
            dtype=np.float32,
        )

        p1 = np.round(midpoint).astype(int)
        p2 = np.round(midpoint + orthogonal_vector).astype(int)
        min_dist = np.sum(np.absolute(p2 - p1))

        # Build shapely geometries from transformed monitor coordinates
        line = LineString([tuple(midpoint), tuple((midpoint + orthogonal_vector))])

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

            other_shape = Polygon(other_outer_points.tolist())

            intersection = line.intersection(other_shape)

            if not intersection.is_empty:
                possible_point = np.round(intersection.coords[0]).astype(int)
                possible_dist = np.sum(np.absolute(possible_point - p1))
                # if there are multiple intersections, choose the
                # one with the lowest manhattan distance
                if possible_dist < min_dist:
                    connections[component_id] = other_component_id
                    p2 = possible_point
                    min_dist = possible_dist

        p1 = tuple(p1)
        p2 = tuple(p2)

        cv2.line(blank_frame, p1, p2, (0, 255, 255), 5)

    return connections


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
