#!/usr/bin/env python3
# Copyright (c) 2026 e-Yantra, IIT Bombay. All rights reserved.
# These simulation files and source code are the intellectual property of e-Yantra,
# IIT Bombay, provided solely for eYRC 2026-27 (Theme: Hola The Explorer).
# Sharing or redistribution of this material, in whole or in part, is not permitted.

'''
*****************************************************************************************
*
*        =============================================
*           Hola The Explorer (HE) Theme (eYRC 2026-27)
*        =============================================
*
*  This script is to implement Task 2B of Hola The Explorer (HE) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''

# Team ID:          3062
# Author List:      Dnyanad Patil, Aruthra B, Nukala Jitendra, Keshav Padia
# Filename:         path_planner.py
# Functions:        astar(), line_clear(), shortcut()
# Global variables: GOAL_DECEL, RAMP_STEPS


############################ WHAT YOU HAVE TO DO ##############################
#
#  r2 must visit the three GOALS in order without touching an obstacle. This
#  node plans its path ONCE and publishes it on /r2/plan; path_follower.py
#  drives it.
#
#    camera frame --frame_to_occupancy()--> grid       (given)
#                 --remove_drivable()-----> grid       (YOURS)
#                 --planning_grid()-------> blocked    (given)
#                 --plan_route()----------> plan       (YOURS)
#                 --to_path_msg()---------> /r2/plan   (YOURS)
#
#  A plan is [(t, x, y), ...]: be at (x, y) t seconds after setting off.
#  To stop on a goal, repeat its point with a later t.
#
#  Frame (/r2/odom): origin at the arena's TOP-LEFT corner, x right, y DOWN.
#
#  Run:  ros2 launch hb_description task2b.launch.py
#        ros2 run task_2b path_planner
#        ros2 run task_2b path_follower
#
###############################################################################

import math

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

################# ADD EXTRA IMPORTS / GLOBALS HERE ############
import heapq

GOAL_DECEL = 0.15                    # m/s^2, how hard the path slows into a goal
RAMP_STEPS = 10                      # poses in each slow-down ramp

###############################################################


# ----------------------------------------------------- arena (given, do not edit)
ROBOTS = ("r2",)
START_CIRCLES = [(0.8969, 2.2616), (1.2192, 2.2616), (1.5415, 2.2616)]
GOALS = [(1.2192, 0.4992),           # visited in this order
         (0.3592, 1.5592),
         (2.0792, 0.6992)]
ARENA_SIZE = 2.4384                  # m
ROBOT_RADIUS = 0.084                 # m
SAFETY_MARGIN = 0.030                # m, kept clear on top of the robot radius

# ----------------------------------------------------- camera (given)
CAMERA_URL = "http://127.0.0.1:8080/stream"
ARENA_LEFT, ARENA_TOP, ARENA_PIXELS = 304, 24, 671       # floor square in the image
GRID_CELLS = 244                     # occupancy grid is GRID_CELLS x GRID_CELLS
CELL_SIZE = ARENA_SIZE / GRID_CELLS  # m, about 1 cm
SAND_LOW, SAND_HIGH = (10, 40, 170), (30, 120, 255)      # sand colour, HSV

# ----------------------------------------------------- planning (tunable)
PLAN_CELL = 0.04                     # m, planning grid cell
PLAN_SPEED = 0.24                    # m/s, the speed the path is timed at
GOAL_STOP = 2.0                      # s, stop this long on each goal
START_DELAY = 5.0                    # s, from publishing to setting off

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


######################### OCCUPANCY GRID ######################

def read_frame(url=CAMERA_URL):
    """Given. One BGR frame from the overhead camera."""
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("no frame from %s -- is task2b.launch.py running?" % url)
    return frame


def crop_arena(frame):
    """Given. The arena floor only, ARENA_PIXELS square."""
    return frame[ARENA_TOP:ARENA_TOP + ARENA_PIXELS, ARENA_LEFT:ARENA_LEFT + ARENA_PIXELS]


def frame_to_occupancy(frame):
    """Given. Frame -> grid[row, col] = grid[y, x], True = obstacle.
    Only sand is free. To view: cv2.imshow("g", np.where(grid, 0, 255).astype(np.uint8))"""
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    obstacles = cv2.bitwise_not(cv2.inRange(hsv, SAND_LOW, SAND_HIGH))
    # drop specks and thin painted lines, then fill the holes inside each obstacle
    obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(obstacles, contours, -1, 255, cv2.FILLED)
    # one pixel per cell; a cell is an obstacle if any part of it is
    small = cv2.resize(obstacles, (GRID_CELLS, GRID_CELLS), interpolation=cv2.INTER_AREA)
    return small > 0


def clear_circle(occupied, xy, radius):
    """Given. Free every cell within `radius` m of point xy (in place)."""
    rows, cols = np.indices(occupied.shape)
    x, y = (cols + 0.5) * CELL_SIZE, (rows + 0.5) * CELL_SIZE
    occupied[(x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2] = False


def remove_drivable(frame, occupied, robots, goals):
    """Free what a robot may drive over; return the grid.

    frame_to_occupancy() marks everything that is not sand as an obstacle --
    including the painted floor, the start circles, the robot and the goal
    dots, which leaves NO path. Keep the pillars (and each pad's payload)."""
    ##############  ADD YOUR CODE HERE  ##############
    # TODO:
    #   - painted floor: find it by colour in crop_arena(frame) and free it
    #     (the pillars are grey, so colour separates them)
    #   - start circles, robots, goal dots: clear_circle(), with room round
    #     each goal for the whole robot
    # Real obstacles (pillars, machines, the payloads' dark rims) are grey or
    # dark; the paint, the pads' rings and the goal dots are saturated colour.
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    sand = cv2.inRange(hsv, SAND_LOW, SAND_HIGH) > 0
    # the vault's light beam and its crystals are pale or bright green, which
    # the grey test would catch; the pillars are colourless and the cyan
    # payload's hue is above 85, so neither is caught here
    beam = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 80) & (hsv[..., 1] >= 20) & (hsv[..., 2] >= 100)
    grey = (((hsv[..., 1] < 60) | (hsv[..., 2] < 60)) & ~sand & ~beam).astype(np.uint8) * 255

    # seal the payloads' rims and fill them in; the hexagon's dashed border
    # closes into a much bigger loop, so only fill payload-sized shapes.
    # Painted dashes and specks are dropped by area.
    grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(grey, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(grey)
    for c in contours:
        _, _, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 300 and max(w, h) < 100:
            cv2.drawContours(solid, [c], -1, 255, cv2.FILLED)
    big = np.zeros_like(grey)
    cv2.drawContours(big, [c for c in contours if max(cv2.boundingRect(c)[2:]) >= 100], -1, 255, cv2.FILLED)
    solid |= grey & big                                    # big loops: the grey pixels only
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    solid = cv2.dilate(solid, np.ones((7, 7), np.uint8))   # cover soft edges and shadows

    keep = cv2.resize(solid, (GRID_CELLS, GRID_CELLS), interpolation=cv2.INTER_AREA) > 0
    occupied = occupied & keep

    for c in START_CIRCLES:
        clear_circle(occupied, c, 0.12)
    for r in robots:
        clear_circle(occupied, r, ROBOT_RADIUS + 0.02)
    for g in goals:   # + a planning cell, so the goal's cell survives planning_grid()
        clear_circle(occupied, g, ROBOT_RADIUS + SAFETY_MARGIN + PLAN_CELL)
    return occupied
    ##################################################


######################### PLANNING GRID #######################

def planning_grid(occupied):
    """Given. Grid -> blocked[i, j] = blocked[x, y] (NOTE: x first) of PLAN_CELL
    cells. Obstacles grow by ROBOT_RADIUS + SAFETY_MARGIN, so plan the robot's
    centre as a point. The arena's edge is a wall."""
    pad = ROBOT_RADIUS + SAFETY_MARGIN
    k = int(math.ceil(pad / CELL_SIZE))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    inflated = cv2.dilate(occupied.astype(np.uint8), disc) > 0

    # a planning cell is blocked if the robot's centre, standing on the cell's
    # centre, would be too close to an obstacle
    n = int(round(ARENA_SIZE / PLAN_CELL))
    centre = (np.arange(n) + 0.5) * PLAN_CELL
    fine = np.minimum((centre / CELL_SIZE).astype(int), GRID_CELLS - 1)
    blocked = inflated[np.ix_(fine, fine)].T        # [row=y, col=x] -> [i=x, j=y]
    near_wall = (centre < pad) | (centre > ARENA_SIZE - pad)
    blocked[near_wall, :] = True
    blocked[:, near_wall] = True
    return blocked


def to_cell(p):
    """Given. Point (x, y) in m -> planning cell (i, j)."""
    n = int(round(ARENA_SIZE / PLAN_CELL))
    return (min(max(int(p[0] / PLAN_CELL), 0), n - 1), min(max(int(p[1] / PLAN_CELL), 0), n - 1))


def to_world(c):
    """Given. Planning cell (i, j) -> its centre (x, y) in m."""
    return ((c[0] + 0.5) * PLAN_CELL, (c[1] + 0.5) * PLAN_CELL)


def nearest_free(blocked, p, max_r=0.30):
    """Given. The free cell closest to point p, or None within max_r m."""
    c = to_cell(p)
    if not blocked[c]:
        return c
    free = [(i, j) for i in range(blocked.shape[0]) for j in range(blocked.shape[1])
            if not blocked[i, j] and abs(i - c[0]) * PLAN_CELL <= max_r
            and abs(j - c[1]) * PLAN_CELL <= max_r]
    return min(free, key=lambda q: math.dist(to_world(q), p)) if free else None


def plan_route(blocked, start):
    """start -> GOALS[0] -> GOALS[1] -> GOALS[2] on `blocked`.
    Returns [(t, x, y), ...] starting with (0.0, *start), or None."""
    ##############  ADD YOUR CODE HERE  ##############
    # TODO:
    #   1. each leg: search the grid between two cells (A* is a good start;
    #      nearest_free() gives a free start/end cell)
    #   2. straighten the zig-zags where the straight line is clear
    #   3. t = distance so far / PLAN_SPEED; on each goal add its point again
    #      at t + GOAL_STOP; end exactly on each goal
    points = [tuple(start)]
    for goal in GOALS:
        a, b = nearest_free(blocked, points[-1]), nearest_free(blocked, goal)
        if a is None or b is None:
            return None
        cells = astar(blocked, a, b)
        if cells is None:
            return None
        points += shortcut(blocked, [to_world(c) for c in cells])[1:-1] + [goal]

    # Time it at PLAN_SPEED, but ease into each goal: the follower cuts the
    # wheels the moment the robot is on the goal, and at full speed the sudden
    # stop rocks it a couple of cm back off the goal.
    route, t = [(0.0, *points[0])], 0.0
    for p, q in zip(points, points[1:]):
        seg = math.dist(p, q)
        if q not in GOALS:
            t += seg / PLAN_SPEED
            route.append((t, *q))
            continue
        ramp = min(seg, PLAN_SPEED ** 2 / (2 * GOAL_DECEL))
        if seg > ramp:                               # cruise up to the ramp
            u = (seg - ramp) / seg
            t += (seg - ramp) / PLAN_SPEED
            route.append((t, p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u))
        for k in range(1, RAMP_STEPS + 1):           # v = sqrt(2 a s) down to a crawl
            s_mid = ramp * (RAMP_STEPS - k + 0.5) / RAMP_STEPS
            v = max(math.sqrt(2 * GOAL_DECEL * s_mid), 0.02)
            t += ramp / RAMP_STEPS / v
            u = 1.0 - ramp * (RAMP_STEPS - k) / RAMP_STEPS / seg
            route.append((t, p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u))
        route[-1] = (t, *q)                          # end exactly on the goal
        t += GOAL_STOP
        route.append((t, *q))
    return route
    ##################################################


def astar(blocked, a, b):
    """A* over free cells, 8-connected, from cell a to cell b. Cells, or None."""
    n_i, n_j = blocked.shape
    steps = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    came, cost, todo = {a: None}, {a: 0.0}, [(math.dist(a, b), a)]
    while todo:
        _, c = heapq.heappop(todo)
        if c == b:
            path = []
            while c is not None:
                path.append(c)
                c = came[c]
            return path[::-1]
        for di, dj, d in steps:
            nb = (c[0] + di, c[1] + dj)
            if not (0 <= nb[0] < n_i and 0 <= nb[1] < n_j) or blocked[nb]:
                continue
            if di and dj and (blocked[c[0] + di, c[1]] or blocked[c[0], c[1] + dj]):
                continue                                # no cutting corners
            g = cost[c] + d
            if g < cost.get(nb, math.inf):
                cost[nb], came[nb] = g, c
                heapq.heappush(todo, (g + math.dist(nb, b), nb))
    return None


def line_clear(blocked, p, q):
    """True if the straight line p -> q (in m) crosses no blocked cell."""
    n = max(1, int(math.dist(p, q) / (PLAN_CELL / 4)))
    return not any(blocked[to_cell((p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n))]
                   for k in range(n + 1))


def shortcut(blocked, pts):
    """Drop the zig-zags: from each point, jump to the farthest one in sight."""
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not line_clear(blocked, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


######################### NODE ################################

class PathPlanner(Node):
    """Given, except to_path_msg(). Waits for /r2/odom, plans once, and
    publishes the path on /r2/plan (latched, so a late follower still gets it)."""

    def __init__(self):
        super().__init__("task_2b_path_planner")
        self.pose, self.pubs = {}, {}
        for r in ROBOTS:
            self.create_subscription(Odometry, f"/{r}/odom", self._odom_cb(r), 10)
            self.pubs[r] = self.create_publisher(Path, f"/{r}/plan", LATCHED)
        self.planned = False
        self.create_timer(0.2, self.tick)
        self.get_logger().info("task 2B planner: %s visits %s" % (ROBOTS[0], GOALS))

    def _odom_cb(self, robot):
        def cb(msg):
            p = msg.pose.pose.position
            self.pose[robot] = (p.x, p.y)
        return cb

    def to_path_msg(self, plan, start):
        """[(t, x, y), ...] and start (rclpy Time) -> nav_msgs/Path."""
        ##############  ADD YOUR CODE HERE  ##############
        # TODO: frame_id "odom" on the path and each PoseStamped; path stamp =
        # start, each pose's stamp = start + Duration(seconds=t) (.to_msg()).
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = start.to_msg()
        for t, x, y in plan:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.header.stamp = (start + Duration(seconds=t)).to_msg()
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg
        ##################################################

    def tick(self):
        if self.planned or len(self.pose) < len(ROBOTS):
            return                              # wait for the robot's odom
        self.planned = True
        robot = ROBOTS[0]

        frame = read_frame()
        occupied = remove_drivable(frame, frame_to_occupancy(frame),
                                   robots=list(self.pose.values()), goals=GOALS)
        blocked = planning_grid(occupied)
        self.get_logger().info("occupancy grid: %.1f%% obstacle; planning grid %d x %d"
                               % (100.0 * occupied.mean(), *blocked.shape))

        route = plan_route(blocked, self.pose[robot])
        if route is None:
            self.get_logger().error("no route. Start and goals, as the planner saw them:")
            for name, p in [("start", self.pose[robot])] + [("goal %d" % (k + 1), g) for k, g in enumerate(GOALS)]:
                self.get_logger().error("  %s %s %s" % (name, p, "BLOCKED" if blocked[to_cell(p)] else "free"))
            return

        start = self.get_clock().now() + Duration(seconds=START_DELAY)
        self.pubs[robot].publish(self.to_path_msg(route, start))
        length = sum(math.dist(a[1:], b[1:]) for a, b in zip(route, route[1:]))
        self.get_logger().info("%s: published /%s/plan, %d poses, %.2f m, finishes at t=%.1f s"
                               % (robot, robot, len(route), length, route[-1][0]))


def main():
    rclpy.init()
    node = PathPlanner()
    try:
        rclpy.spin(node)                        # keeps the latched path available
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
