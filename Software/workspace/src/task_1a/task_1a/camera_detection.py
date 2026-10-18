#!/usr/bin/env python3
##!/usr/bin/env python3
# Copyright (c) 2026 e-Yantra, IIT Bombay. All rights reserved.
# These simulation files and source code are the intellectual property of e-Yantra,
# IIT Bombay, provided solely for eYRC 2026-27 (Theme: Hola The Explorer).
# Sharing or redistribution of this material, in whole or in part, is not permitted.

'''
*****************************************************************************************
*
*        =============================================
*           Hola The Explorer (HE) Theme (eYRC 2025-26)
*        =============================================
*
*  This script is to implement Task 1A of Hola The Explorer (HE) Theme (eYRC 2025-26).
*
*****************************************************************************************
'''

# Team ID:          3062
# Author List:      Aruthra B
# Filename:         camera_detection.py
# Functions:        centre_of_quad(), find_trapezoids(), main()
# Global variables: STREAM_URL, WINDOW, BINARY_WINDOW, FPS_WINDOW, ARENA_*, SAND_DISTANCE,
#                   HOUGH_*, MIN_TRAPEZOID_AREA, PARALLEL_TOLERANCE_DEG, REPORT_PERIOD_SEC
# Service Clients:  pixel_to_world  ->  shape_interface/srv/PixelToWorld

import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from shape_interface.srv import PixelToWorld

##################### TUNABLE CONSTANTS #######################
STREAM_URL = "http://127.0.0.1:8080/stream"
WINDOW = "camera_feed"
BINARY_WINDOW = "trapezoid_borders"
FPS_WINDOW = 30

ARENA_X0, ARENA_Y0, ARENA_X1, ARENA_Y1 = 304, 24, 975, 695
SAND_DISTANCE = 18

HOUGH_THRESHOLD = 40
HOUGH_MIN_LENGTH = 35
HOUGH_MAX_GAP = 25

MIN_TRAPEZOID_AREA = 1200
PARALLEL_TOLERANCE_DEG = 7
REPORT_PERIOD_SEC = 0.5
###############################################################


def centre_of_quad(corners):
    """
    Find the centroid of a quadrilateral using image moments.
    """
    pts = np.asarray(corners, dtype=np.float32).reshape((-1, 1, 2))
    M = cv2.moments(pts)
    if abs(M['m00']) > 1e-5:
        cx = float(M['m10'] / M['m00'])
        cy = float(M['m01'] / M['m00'])
    else:
        cx = float(np.mean(pts[:, 0, 0]))
        cy = float(np.mean(pts[:, 0, 1]))
    return cx, cy

def find_trapezoids(frame):
    h, w = frame.shape[:2]
    binary = np.zeros((h, w), dtype=np.uint8)
    trapezoids = []

    # 1. Crop to arena region
    x0, y0, x1, y1 = ARENA_X0, ARENA_Y0, ARENA_X1, ARENA_Y1
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return binary, trapezoids

    # 2. LAB floor distance segmentation
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    floor_color = np.array([
        np.bincount(lab[:, :, 0].flatten()).argmax(),
        np.bincount(lab[:, :, 1].flatten()).argmax(),
        np.bincount(lab[:, :, 2].flatten()).argmax()
    ], dtype=np.float32)

    diff = np.linalg.norm(lab.astype(np.float32) - floor_color, axis=2)
    mask = (diff > SAND_DISTANCE).astype(np.uint8) * 255

    # 3. Detect line segments with Canny + HoughLinesP
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LENGTH, maxLineGap=HOUGH_MAX_GAP)

    scratch = np.zeros(crop.shape[:2], dtype=np.uint8)
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            cv2.line(scratch, (lx1, ly1), (lx2, ly2), 255, 4)

    # Bridge gaps so borders form closed loops
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    scratch = cv2.morphologyEx(scratch, cv2.MORPH_CLOSE, close_kernel)

    # 4. Find enclosed regions
    contours, hierarchy = cv2.findContours(scratch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    accepted_corners = []

    if hierarchy is not None:
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < MIN_TRAPEZOID_AREA or area > 70000:
                continue

            epsilon = 0.035 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2)

                angles = []
                for i in range(4):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % 4]
                    ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0
                    angles.append(ang)

                def is_parallel(a, b, tol=PARALLEL_TOLERANCE_DEG):
                    d = abs(a - b)
                    return d < tol or abs(d - 180.0) < tol

                parallel_pairs = 0
                if is_parallel(angles[0], angles[2]):
                    parallel_pairs += 1
                if is_parallel(angles[1], angles[3]):
                    parallel_pairs += 1

                # Exactly 1 pair of parallel sides = trapezoid
                if parallel_pairs == 1:
                    full_corners = pts.astype(np.float32) + np.array([x0, y0], dtype=np.float32)
                    cx, cy = centre_of_quad(full_corners)
                    trapezoids.append((cx, cy, full_corners))
                    accepted_corners.append(np.round(full_corners).astype(np.int32))

    # 5. Draw white outlines (thickness 2) on the binary mask
    if accepted_corners:
        cv2.drawContours(binary, accepted_corners, -1, 255, thickness=2)

    if len(trapezoids) == 3:
        cv2.imwrite("HE_3062_binary.png", binary)

    return binary, trapezoids


def main():
    rclpy.init()
    node = Node("camera_feed")
    client = node.create_client(PixelToWorld, "pixel_to_world")

    node.get_logger().info("waiting for the pixel_to_world service ...")
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error(
            "pixel_to_world is not up. Start it first: ros2 run task_1a pixel_to_world_service")
        rclpy.shutdown()
        return

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        node.get_logger().error(f"could not open {STREAM_URL}")
        node.get_logger().error(
            "start the simulation first: ros2 launch hb_description task1a.launch.py")
        rclpy.shutdown()
        return

    stamps = deque(maxlen=FPS_WINDOW)
    fps = 0.0
    last_report = 0.0

    while rclpy.ok():
        ok, frame = cap.read()
        if not ok:
            break

        stamps.append(time.monotonic())
        if len(stamps) >= 2:
            span = stamps[-1] - stamps[0]
            fps = (len(stamps) - 1) / span if span > 0 else 0.0

        binary, trapezoids = find_trapezoids(frame)

        for cx, cy, corners in trapezoids:
            cv2.polylines(frame, [np.round(corners).astype(np.int32)], True, (0, 0, 255), 2)
            cv2.circle(frame, (int(round(cx)), int(round(cy))), 6, (0, 255, 255), -1)

        cv2.putText(frame, f"{fps:5.1f} FPS", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{len(trapezoids)} trapezoids", (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)
        cv2.imshow(BINARY_WINDOW, binary)

        now = time.monotonic()
        if trapezoids and now - last_report >= REPORT_PERIOD_SEC:
            last_report = now
            print(f"\n{len(trapezoids)} trapezoid(s):")

            for cx, cy, _ in sorted(trapezoids, key=lambda t: (t[1], t[0])):
                req = PixelToWorld.Request()
                req.pixel_x = float(cx)
                req.pixel_y = float(cy)

                future = client.call_async(req)
                rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)

                result = future.result()
                if result is None:
                    print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  timed out")
                elif not result.success:
                    print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  failed: {result.message}")
                else:
                    print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  world ({result.world_x:6.3f}, {result.world_y:6.3f}) m")

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
