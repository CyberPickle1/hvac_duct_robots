# ROS and message type imports
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile

# Scientific / image processing
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import open3d as o3d

# Utilities
import time
import math
from statistics import median
import numpy as np
from collections import defaultdict
from matplotlib.path import Path as MplPath

# CUDA
import cupy as cp
from open3d.core import Device, Tensor

from pypackage.precomputed_path import precomputed_path
from pypackage.occupancy_grid_util import occupnacy_grid


class Navigator(Node):
    def __init__(self):
        super().__init__('test_bed')
        qos = QoSProfile(depth=10)

        # --- Subscribers ---
        self.subscription_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.subscription_camera = self.create_subscription(
            PointCloud2, '/camera/depth_camera_sensor/points', self.camera_callback, 10)
        self.subscription_wall = self.create_subscription(
            PointCloud2, '/wall_point_cloud', self.wall_callback, 10)

        # message_filters subscribers for time synchronisation
        self.odom_sub_sync = Subscriber(self, Odometry, '/odom', qos_profile=qos)
        self.camera_sub_sync = Subscriber(
            self, PointCloud2, '/camera/depth_camera_sensor/points', qos_profile=qos)

        # --- Publishers ---
        self.publisher       = self.create_publisher(Twist,       'cmd_vel',                10)
        self.pc_pub          = self.create_publisher(PointCloud2, '/clean_point_cloud',     10)
        self.stitched_pub    = self.create_publisher(PointCloud2, '/stitched_point_cloud',  10)
        self.wall_pub        = self.create_publisher(PointCloud2, '/wall_point_cloud',      10)
        self.better_marker_pub = self.create_publisher(MarkerArray, 'better_targets_markers', 10)
        self.path_pub        = self.create_publisher(Path,        'planned_path',           10)
        self.marker_pub      = self.create_publisher(MarkerArray, 'robot_pose_marker',      10)
        self.marker_pubs     = self.create_publisher(MarkerArray, '/projected_pose_marker', 10)
        self.raw_pc_pub      = self.create_publisher(PointCloud2, 'raw_cloud',              10)

        # --- Time synchroniser ---
        self.time_sync = ApproximateTimeSynchronizer(
            [self.odom_sub_sync, self.camera_sub_sync],
            queue_size=10,
            slop=0.005,
        )
        self.time_sync.registerCallback(self.SyncCallback)

        # --- Occupancy grids & skeleton maps ---
        self.grid1, self.origin  = occupnacy_grid(0.05)
        self.grid2, self.origin2 = occupnacy_grid(0.05)

        self.skel_coords1, self.spacing1 = self.get_skeleton_map_and_spacing(self.grid1,  self.origin)
        self.skel_coords2, self.spacing2 = self.get_skeleton_map_and_spacing(self.grid2, self.origin2)

        # --- Odometry origin offset ---
        self.odom_offset_x =  0.0
        self.odom_offset_y = -0.048202

        # --- Floor targets ---
        floor_targets = {
            0: [
                (349, 188), (448, 190), (449,  29),
                (348,  10), ( 72,  30),
                ( 18,  28), (110, 230), ( 20, 235),
                ( 22, 108), ( 47, 248), (427, 244),
                (427, 431), (197, 431), (170, 437),
                ( 39, 437),
            ]
        }

        raw_path = precomputed_path
        reordered_targets = self.reorder_targets_indices(raw_path, floor_targets)

        self.floor_targets = {
            f: self.indices_to_meters(pts, self.origin if f == 0 else self.origin2)
            for f, pts in reordered_targets.items()
        }

        # --- Path / goals ---
        self.path  = self.indices_to_meters(raw_path, self.origin)
        self.goals = self.downsample_path(self.path, step=5)

        # --- Navigation tolerances ---
        self.distance_tolerance    = 0.1
        self.goal_distance         = 0.0
        self.goal_orientation      = 0.0
        self.orientation_tolerance = math.radians(5)

        # --- Control outputs ---
        self.vx = 0.0
        self.wz = 0.0

        # --- State ---
        self.i                 = 0
        self.narrow_entry      = None
        self.backing_out       = False
        self.is_reached        = False
        self.just_reached      = False
        self.floor_targets_idx = 0
        self.in_narrow         = False
        self.rotating          = False
        self.in_avoidance_phase       = False
        self.avoidance_risk_threshold = 0.5

        # --- Pose ---
        self.x             = 0.0
        self.y             = 0.0
        self.z             = 0.0
        self.theta_current = 0.0
        self.path_index    = 0
        self.cam_pose_ready    = False
        self.last_robot_quat   = None
        self.last_robot_pos    = None

        # --- Point clouds ---
        self.latest_clean_cloud = o3d.geometry.PointCloud()
        self.stitched_cloud     = o3d.geometry.PointCloud()
        self.latest_wall_cloud  = np.empty((0, 3))
        self.wall_points        = np.empty((0, 2))
        self.max_stitch_radius  = 5.0

        # Camera offset: camera_link -> chassis_link translation
        self.cam_translation = np.array([0.0, -0.05, 0.0])

        # --- PD controller state dicts ---
        self.linear_pd_state  = {}
        self.angular_pd_state = {}

        self.last_log_time = 0.0

        # --- CUDA device ---
        self._cuda_device = Device("CUDA:0")

        print(self.floor_targets)

    # ------------------------------------------------------------------ #
    #  Path helpers                                                        #
    # ------------------------------------------------------------------ #

    def reorder_targets_indices(self, path, targets, floor=0, i=0):
        empty_dict   = {}
        floors_left  = set(targets.keys())

        for point in path:
            if not floors_left:
                break

            point_floor, row, col = point

            if point_floor != floor:
                continue

            if (row, col) in targets.get(floor, []):
                if floor not in empty_dict:
                    empty_dict[floor] = []
                if (row, col) not in empty_dict[floor]:
                    empty_dict[floor].append((floor, row, col))
                    i += 1

                if i == len(targets.get(floor, [])):
                    floors_left.discard(floor)
                    floor += 1
                    i = 0

        return empty_dict

    def indices_to_meters(self, path, bounds_min, pitch=0.05):
        x_min, y_min = bounds_min
        return [
            (floor, x_min + row * pitch, y_min + col * pitch)
            for (floor, row, col) in path
        ]

    def downsample_path(self, path, step=5):
        if len(path) == 0:
            return path
        downsampled = path[::step]
        if path[-1] != downsampled[-1]:
            downsampled.append(path[-1])
        return downsampled

    def reached(self, x, y, target, tol=0.1):
        return math.sqrt((x - target[1])**2 + (y - target[2])**2) <= tol

    # ------------------------------------------------------------------ #
    #  Skeleton / narrow-section helpers                                   #
    # ------------------------------------------------------------------ #

    def get_skeleton_map_and_spacing(self, grid, bounds_min, pitch=0.05):
        binary   = grid.astype(bool)
        skeleton = skeletonize(binary).astype(int)

        distance_map   = distance_transform_edt(binary)
        spacing_values = distance_map[skeleton == 1] * 2 * pitch

        skeleton_indices     = np.column_stack(np.where(skeleton == 1))
        x_min, y_min         = bounds_min
        skeleton_world_coords = np.array([
            [x_min + r * pitch, y_min + c * pitch]
            for r, c in skeleton_indices
        ])
        return skeleton_world_coords, spacing_values

    def is_in_narrow_section(self, robot_x, robot_y, floor, threshold=0.4):
        coords  = self.skel_coords1 if floor == 0 else self.skel_coords2
        spacing = self.spacing1     if floor == 0 else self.spacing2

        dists     = np.linalg.norm(coords - np.array([robot_x, robot_y]), axis=1)
        min_index = np.argmin(dists)
        min_dist  = dists[min_index]

        if min_dist < 0.25:
            return spacing[min_index] < threshold
        return False

    # ------------------------------------------------------------------ #
    #  Point-cloud utilities                                               #
    # ------------------------------------------------------------------ #

    def pc_creator(self, msg, pose):
        if not self.cam_pose_ready:
            self.get_logger().warn("Camera pose not ready yet, skipping point cloud creation")
            return o3d.t.geometry.PointCloud(
                Tensor(np.empty((0, 3), dtype=np.float32), device=self._cuda_device)
            )

        max_distance_sq = 3.0 ** 2
        pts = [
            (x, y, z)
            for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
            and (x*x + y*y + z*z) <= max_distance_sq
        ]

        if not pts:
            return o3d.t.geometry.PointCloud(
                Tensor(np.empty((0, 3), dtype=np.float32), device=self._cuda_device)
            )

        pts_np = np.asarray(pts, dtype=np.float32)
        P_gpu  = cp.asarray(pts_np)

        # Step 1: camera_depth_optical_frame -> camera_link
        r1_mat = R.from_euler('xyz', [-np.pi / 2, 0.0, -np.pi / 2]).as_matrix().astype(np.float32)
        P_gpu  = P_gpu @ cp.asarray(r1_mat).T

        # Step 2: camera_link -> chassis_link (translation only)
        cam_offset = np.array([0.0, 0.05, 0.0], dtype=np.float32)
        P_gpu += cp.asarray(cam_offset)

        # Step 3: chassis_link -> odom (robot rotation + translation)
        quat       = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        r_robot    = R.from_quat(quat).as_matrix().astype(np.float32)
        pos_gpu    = cp.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
        P_gpu      = P_gpu @ cp.asarray(r_robot).T + pos_gpu

        return o3d.t.geometry.PointCloud(
            Tensor(cp.asnumpy(P_gpu), device=self._cuda_device)
        )

    def clean_point_cloud(self, cloud_gpu):
        if isinstance(cloud_gpu, o3d.geometry.PointCloud):
            arr = np.asarray(cloud_gpu.points, dtype=np.float32)
            if arr.size == 0:
                return o3d.t.geometry.PointCloud(
                    Tensor(np.empty((0, 3), dtype=np.float32), device=self._cuda_device)
                )
            cloud_gpu = o3d.t.geometry.PointCloud(
                Tensor(cp.asnumpy(cp.asarray(arr)), device=self._cuda_device)
            )

        try:
            if cloud_gpu.point.positions.shape[0] == 0:
                return cloud_gpu
        except Exception:
            return cloud_gpu

        try:
            return cloud_gpu.voxel_down_sample(0.02)
        except Exception as e:
            self.get_logger().warn(f"GPU voxel_down_sample failed: {e}")
            pts     = self._gpu_point_positions_to_numpy(cloud_gpu)
            pc_cpu  = o3d.geometry.PointCloud()
            pc_cpu.points = o3d.utility.Vector3dVector(pts)
            cpu_down = pc_cpu.voxel_down_sample(0.02)
            return self._gpu_pointcloud_from_numpy(np.asarray(cpu_down.points))

    def _gpu_pointcloud_from_numpy(self, points_np: np.ndarray):
        if points_np.size == 0:
            return o3d.t.geometry.PointCloud(
                Tensor(np.empty((0, 3), dtype=np.float32), device=self._cuda_device)
            )
        P_gpu = cp.asarray(points_np.astype(np.float32))
        return o3d.t.geometry.PointCloud(
            Tensor(cp.asnumpy(P_gpu), device=self._cuda_device)
        )

    def _gpu_point_positions_to_numpy(self, pc_gpu):
        try:
            pos_tensor = pc_gpu.point.positions
        except Exception:
            return np.asarray(pc_gpu.points)
        return np.array(pos_tensor.cpu().numpy(), dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Collision / visibility / target selection                           #
    # ------------------------------------------------------------------ #

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def check_footprint_collision(self, candidate_pose, heading, point_cloud, robot_shape):
        cx, cy = candidate_pose
        rot = np.array([
            [math.cos(heading), -math.sin(heading)],
            [math.sin(heading),  math.cos(heading)],
        ])
        transformed  = (rot @ robot_shape.T).T + np.array([cx, cy])
        poly         = MplPath(transformed, closed=True)
        inside_ratio = np.count_nonzero(poly.contains_points(point_cloud)) / len(point_cloud)
        return inside_ratio > 0.01

    def build_visibility_map(self, x, y, theta, point_cloud,
                              angle_resolution=np.deg2rad(2), min_points_per_bucket=3):
        angle_buckets = defaultdict(list)
        for px, py in point_cloud:
            dx, dy = px - x, py - y
            r = math.hypot(dx, dy)
            a = self.normalize_angle(math.atan2(dy, dx) - theta)
            bucket_angle = round(a / angle_resolution) * angle_resolution
            angle_buckets[bucket_angle].append(r)

        visibility_map = {}
        for a, distances in angle_buckets.items():
            if not distances:
                visibility_map[a] = None
            elif len(distances) < min_points_per_bucket:
                visibility_map[a] = median(distances)
            else:
                visibility_map[a] = min(distances)

        sorted_angles = sorted(visibility_map)
        for i, a in enumerate(sorted_angles):
            if visibility_map[a] is None:
                left  = visibility_map[sorted_angles[i - 1]]     if i > 0                          else None
                right = visibility_map[sorted_angles[i + 1]] if i + 1 < len(sorted_angles) else None
                neighbors = [v for v in (left, right) if v is not None]
                if neighbors:
                    visibility_map[a] = min(neighbors)

        return visibility_map

    def path_predictor(self, x, y, theta, tol_linear, tol_angular, path, index, size, point_cloud):
        _, x_prime, y_prime = path[index]
        r_expected  = math.hypot(x_prime - x, y_prime - y)
        bounds_linear_max = r_expected + tol_linear
        bounds_linear_min = max(0.01, r_expected - tol_linear)

        angle_to_tgt      = math.atan2(y_prime - y, x_prime - x) - theta
        bounds_angular_max = angle_to_tgt + tol_angular
        bounds_angular_min = angle_to_tgt - tol_angular

        sigma_r     = tol_linear / 2
        sigma_theta = tol_angular / 2
        step_r      = 0.05
        step_a      = np.deg2rad(2)

        r_values        = np.arange(bounds_linear_min, bounds_linear_max, step_r)
        a_values        = np.arange(bounds_angular_min, bounds_angular_max, step_a)
        visibility_map  = self.build_visibility_map(x, y, theta, point_cloud)

        region_points = []
        probabilities = []

        for r in r_values:
            for a in a_values:
                global_angle = theta + a
                px = x + r * math.cos(global_angle)
                py = y + r * math.sin(global_angle)

                ray_limit = visibility_map.get(round(a / step_a) * step_a, float('inf'))
                if ray_limit is not None:
                    if ray_limit == 0.0 or r > ray_limit:
                        continue
                    if r <= bounds_linear_min + 0.1:
                        continue

                pr = math.exp(-((r - r_expected) ** 2) / (2 * sigma_r ** 2))
                pa = math.exp(-(a ** 2) / (2 * sigma_theta ** 2))
                region_points.append((px, py))
                probabilities.append(pr * pa)

        points_in_region = [
            (px, py)
            for px, py in point_cloud
            if math.hypot(px - x, py - y) <= 3.0
            and bounds_linear_min <= math.hypot(px - x, py - y) <= bounds_linear_max
            and bounds_angular_min <= self.normalize_angle(math.atan2(py - y, px - x) - theta) <= bounds_angular_max
        ]

        return region_points, probabilities, points_in_region

    def collision_risk(self, candidate, point_cloud, sigma_w=0.1):
        if point_cloud.size == 0:
            return 0
        cx, cy  = candidate
        d_min   = min(math.hypot(cx - wx, cy - wy) for wx, wy in point_cloud)
        return math.exp(-(d_min ** 2) / (2 * sigma_w ** 2))

    def pick_better_target(self, x, y, theta, tol_linear, tol_angular,
                            path, index, size, point_cloud_raw, reverse=False):
        point_cloud_xy = np.array([[p[0], p[1]] for p in point_cloud_raw])
        if len(point_cloud_xy) == 0:
            return None, None, [], []

        if reverse:
            rot      = np.array([[math.cos(theta), -math.sin(theta)],
                                  [math.sin(theta),  math.cos(theta)]])
            rear_ref = np.array([x, y]) + rot @ np.array([-0.09, 0.0])
            x, y     = rear_ref

        now           = time.time()
        debug_enabled = now - self.last_log_time > 0.5
        if debug_enabled:
            self.last_log_time = now

        gainz = 0.015
        robot_shape = np.array([
            [ 0.0,           0.05 + gainz],
            [ 0.0,          -0.05 - gainz],
            [-0.13 - gainz, -0.05 - gainz],
            [-0.13 - gainz,  0.05 + gainz],
        ])

        region_points, probabilities, _ = self.path_predictor(
            x, y, theta, tol_linear, tol_angular, path, index, size, point_cloud_xy)

        if 0 < index < len(path):
            _, nx, ny   = path[index]
            _, pr_x, pr_y = path[index - 1]
            next_dir = math.atan2(ny - pr_y, nx - pr_x)
        else:
            next_dir = theta

        best_cost  = float('inf')
        best_point = None
        best_info  = None
        was_shifted = False

        max_shift_radius = 0.25
        num_directions   = 30
        angle_step       = math.pi / num_directions

        for (px, py), p in zip(region_points, probabilities):
            risk = self.collision_risk((px, py), point_cloud_xy)

            if p >= 0.05 and not self.check_footprint_collision(
                    (px, py), next_dir, point_cloud_xy, robot_shape):
                collision_risk = p * risk
                if collision_risk < 0.1:
                    continue
                if collision_risk < 0.01:
                    best_point  = (px, py)
                    best_info   = (p, 0.0, risk, collision_risk)
                    was_shifted = False
                    break

            found_safe       = False
            best_shift_cost  = float('inf')
            best_shift_point = None
            best_shift_info  = None

            for i in range(-num_directions // 2, num_directions // 2 + 1):
                search_dir = next_dir + i * angle_step
                shifted_px = px + 0.05 * math.cos(search_dir)
                shifted_py = py + 0.05 * math.sin(search_dir)

                if math.hypot(shifted_px - px, shifted_py - py) > max_shift_radius:
                    continue
                if self.check_footprint_collision(
                        (shifted_px, shifted_py), next_dir, point_cloud_xy, robot_shape):
                    continue

                shifted_angle_diff = abs(self.normalize_angle(
                    math.atan2(shifted_py - y, shifted_px - x) - next_dir))
                shifted_risk = self.collision_risk((shifted_px, shifted_py), point_cloud_xy)

                if self.in_narrow and shifted_angle_diff > np.deg2rad(2):
                    continue
                if p < 0.05:
                    continue

                cost = (1.2 * shifted_angle_diff + 3.0 * shifted_risk + 1.0 * math.hypot(
                    shifted_px - px, shifted_py - py)) * (1.0 - p)

                if cost < best_shift_cost:
                    best_shift_cost  = cost
                    best_shift_point = (shifted_px, shifted_py)
                    best_shift_info  = (p, shifted_angle_diff, shifted_risk, cost)
                    found_safe       = True

            if found_safe and best_shift_cost < best_cost:
                best_cost   = best_shift_cost
                best_point  = best_shift_point
                best_info   = best_shift_info
                was_shifted = True

            if best_point is not None and was_shifted and best_info[3] >= 0.5:
                forward_dx = math.cos(theta)
                forward_dy = math.sin(theta)
                gx, gy = path[index][1], path[index][2]
                if index + 1 < len(path):
                    nx, ny = path[index + 1][1], path[index + 1][2]
                else:
                    nx, ny = gx, gy

                proj_current = (gx - x) * forward_dx + (gy - y) * forward_dy
                proj_next    = (nx - x) * forward_dx + (ny - y) * forward_dy

                if proj_current <= proj_next:
                    for offset in range(1, 4):
                        next_idx = index + offset
                        if next_idx >= len(path):
                            break
                        _, fx, fy = path[next_idx]
                        if self.check_footprint_collision(
                                (fx, fy), next_dir, point_cloud_xy, robot_shape):
                            shift_dx = best_point[0] - px
                            shift_dy = best_point[1] - py
                            path[next_idx] = (path[next_idx][0], fx + shift_dx, fy + shift_dy)
                        else:
                            break

        return best_point, best_info, region_points, probabilities

    def pd_control(self, error, state, kp, kd):
        now = time.time()
        dt  = now - state.get('last_time', now)
        if dt == 0:
            dt = 1e-6

        d_error = (error - state.get('previous_error', 0)) / dt
        output  = kp * error + kd * d_error

        state['previous_error'] = error
        state['last_time']      = now
        return output

    # ------------------------------------------------------------------ #
    #  Publishers                                                          #
    # ------------------------------------------------------------------ #

    def publish_path(self):
        msg             = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp    = self.get_clock().now().to_msg()

        for goal in self.goals:
            pose                       = PoseStamped()
            pose.header.frame_id       = "odom"
            pose.pose.position.x       = goal[1]
            pose.pose.position.y       = goal[2]
            pose.pose.position.z       = self.z
            pose.pose.orientation.w    = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def publish_odometry(self):
        marker_array = MarkerArray()

        # Sphere: robot position
        sphere              = Marker()
        sphere.header.frame_id = "odom"
        sphere.header.stamp    = self.get_clock().now().to_msg()
        sphere.ns              = "robot_pose"
        sphere.id              = 0
        sphere.type            = Marker.SPHERE
        sphere.action          = Marker.ADD
        sphere.pose.position.x = self.x
        sphere.pose.position.y = self.y
        sphere.pose.position.z = self.z
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 0.13
        sphere.scale.y = 0.2
        sphere.scale.z = 0.2
        sphere.color.r = 1.0
        sphere.color.a = 1.0
        marker_array.markers.append(sphere)

        # Arrow: heading direction
        arrow              = Marker()
        arrow.header.frame_id = "odom"
        arrow.header.stamp    = self.get_clock().now().to_msg()
        arrow.ns               = "robot_heading"
        arrow.id               = 1
        arrow.type             = Marker.ARROW
        arrow.action           = Marker.ADD
        arrow.points = [
            Point(x=self.x, y=self.y, z=self.z),
            Point(x=self.x + 0.3 * math.cos(self.theta_current),
                  y=self.y + 0.3 * math.sin(self.theta_current),
                  z=self.z),
        ]
        arrow.scale.x = 0.05
        arrow.scale.y = 0.1
        arrow.scale.z = 0.1
        arrow.color.r = 1.0
        arrow.color.a = 1.0
        marker_array.markers.append(arrow)

        self.marker_pub.publish(marker_array)

    def publish_better_target_marker(self, point):
        marker                  = Marker()
        marker.header.frame_id  = "odom"
        marker.header.stamp     = self.get_clock().now().to_msg()
        marker.ns               = "better_target"
        marker.id               = 0
        marker.type             = Marker.SPHERE
        marker.action           = Marker.ADD
        marker.pose.position.x  = point[0]
        marker.pose.position.y  = point[1]
        marker.pose.position.z  = self.z
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r = 1.0
        marker.color.a = 1.0

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.better_marker_pub.publish(marker_array)

    def publish_projected_direction(self, avoid_dir):
        marker                 = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns              = "projected"
        marker.id              = 0
        marker.type            = Marker.ARROW
        marker.action          = Marker.ADD
        marker.points = [
            Point(x=self.x, y=self.y, z=self.z),
            Point(x=self.x + avoid_dir[0] * 0.5,
                  y=self.y + avoid_dir[1] * 0.5,
                  z=self.z),
        ]
        marker.scale.x = 0.05
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.a = 1.0
        marker.lifetime.sec = 1

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_pubs.publish(marker_array)

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def SyncCallback(self, odom_msg, camera_msg):
        self.odom_callback(odom_msg)
        self.camera_callback(camera_msg, pose=odom_msg.pose.pose)

    def wall_callback(self, msg):
        cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        if not cloud_points:
            self.latest_wall_cloud = np.empty((0, 3))
            return

        pts = np.array([[p[0], p[1], p[2]] for p in cloud_points], dtype=np.float64)

        if pts.ndim == 1 and pts.size == 3:
            pts = pts.reshape(1, 3)

        if pts.ndim == 2 and pts.shape[1] == 3:
            dists = np.linalg.norm(pts - np.array([self.x, self.y, self.z]), axis=1)
            self.latest_wall_cloud = pts[dists <= 0.5]
        else:
            self.latest_wall_cloud = np.empty((0, 3))

    def odom_callback(self, msg):
        qx, qy, qz, qw = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        self.robot_quat = [qx, qy, qz, qw]

        if self.i >= len(self.goals):
            self.vx = 0.0
            self.wz = 0.0
            print("All goals reached!")
            return

        self.goal          = self.goals[self.i]
        self.theta_current = R.from_quat([qx, qy, qz, qw]).as_euler('zyx')[0]

        pos     = np.array([msg.pose.pose.position.x,
                            msg.pose.pose.position.y,
                            msg.pose.pose.position.z], dtype=np.float32)
        rot_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()

        global_offset    = rot_mat @ np.array([self.odom_offset_x, self.odom_offset_y, 0.0], dtype=np.float32)
        self.x, self.y, self.z = pos - global_offset

        x_goal, y_goal = self.goal[1], self.goal[2]
        self.goal_distance = math.sqrt((x_goal - self.x)**2 + (y_goal - self.y)**2)

        rpy_goal = math.atan2(self.goal[2] - self.y, self.goal[1] - self.x) - self.theta_current
        rpy_goal = (rpy_goal + math.pi) % (2 * math.pi) - math.pi

        goal_reached    = self.goal_distance < self.distance_tolerance
        needs_rotation  = abs(rpy_goal) > self.orientation_tolerance
        can_move_forward = not needs_rotation and self.goal_distance > self.distance_tolerance

        floor            = self.goal[0]
        targets_on_floor = self.floor_targets[floor]

        # --- Floor-target check ---
        if not self.just_reached and self.floor_targets_idx < len(targets_on_floor):
            tgt             = targets_on_floor[self.floor_targets_idx]
            self.is_reached = self.reached(self.x, self.y, tgt)
            if self.is_reached:
                self.just_reached = True
                self.reach_time   = time.time()

        if self.just_reached and time.time() - self.reach_time > 5.0:
            self.floor_targets_idx += 1
            self.just_reached = False
            self.is_reached   = False
            print("Next target")

        # --- Narrow-section detection ---
        in_narrow    = self.is_in_narrow_section(self.x, self.y, floor)
        self.in_narrow = in_narrow

        if self.is_reached and needs_rotation and in_narrow and not self.backing_out:
            self.backing_out = True
            print("Dead end in narrow duct – switching to reverse")

        # --- Better-target selection ---
        if len(self.latest_clean_cloud.points) > 0:
            best_point, best_info, _, _ = self.pick_better_target(
                x=self.x, y=self.y, theta=self.theta_current,
                tol_linear=0.1, tol_angular=np.deg2rad(5),
                path=self.goals, index=self.i, size=len(self.goals),
                point_cloud_raw=self.latest_wall_cloud,
                reverse=self.backing_out,
            )
            if best_point is not None:
                self.get_logger().info(f"Better target: {best_point}  info: {best_info}")
                self.publish_better_target_marker(best_point)

        # --- Motion commands ---
        if goal_reached:
            self.i += 1
            print(f"Goal reached! Moving to index {self.i}")

        elif self.backing_out:
            reverse_shift = 0.09
            x_center      = self.x - reverse_shift

            rpy_rev_goal = math.atan2(self.y - self.goal[2], x_center - self.goal[1]) - self.theta_current
            rpy_rev_goal = (rpy_rev_goal + math.pi) % (2 * math.pi) - math.pi
            if abs(rpy_goal) < math.radians(5):
                rpy_goal = 0.0

            rev_aligned = abs(rpy_rev_goal) < self.orientation_tolerance
            self.get_logger().info(
                f"[REVERSE] rpy_rev_goal: {rpy_rev_goal:.2f}  aligned: {rev_aligned}")

            if not rev_aligned:
                self.vx = 0.0
                self.wz = 1.0 * np.sign(rpy_rev_goal)
            else:
                self.vx = -0.5
                self.wz =  0.0

            if not in_narrow:
                self.backing_out = False
                self.vx = 0.0
                self.wz = 0.0
                print("Exited narrow duct – resuming forward")

        elif needs_rotation:
            self.vx = 0.0
            self.wz = 1.0 * np.sign(rpy_goal)

        elif can_move_forward:
            self.vx = 0.5
            self.wz = 0.0

        else:
            self.vx = 0.0
            self.wz = 0.0

        self.rotating = needs_rotation and not self.backing_out

        cmd             = Twist()
        cmd.linear.x    = self.vx
        cmd.angular.z   = self.wz
        self.publisher.publish(cmd)

        print(f"Pose: ({self.x:.3f}, {self.y:.3f}) | "
              f"Dist: {self.goal_distance:.3f} | "
              f"rpy_goal: {math.degrees(rpy_goal):.2f}° | "
              f"theta: {math.degrees(self.theta_current):.2f}°")

        self.cam_pose_ready = True
        self.publish_odometry()
        self.publish_path()

    def camera_callback(self, msg, pose=None):
        if self.x == 0.0 and self.y == 0.0 and self.theta_current == 0.0:
            return

        self.raw_pc_pub.publish(msg)

        if pose is None:
            return

        # Build and clean GPU point cloud
        new_cloud_gpu   = self.pc_creator(msg, pose)
        clean_cloud_gpu = self.clean_point_cloud(new_cloud_gpu)
        clean_array     = self._gpu_point_positions_to_numpy(clean_cloud_gpu)

        # Update CPU copy for the rest of the pipeline
        if clean_array.shape[0] > 0:
            pc_cpu        = o3d.geometry.PointCloud()
            pc_cpu.points = o3d.utility.Vector3dVector(clean_array)
            self.latest_clean_cloud = pc_cpu
        else:
            self.latest_clean_cloud = o3d.geometry.PointCloud()

        # Accumulate stitched cloud and cull by radius
        stitched_pts = np.asarray(self.stitched_cloud.points)
        if stitched_pts.size == 0:
            stitched_pts = clean_array
        elif clean_array.size > 0:
            stitched_pts = np.vstack((stitched_pts, clean_array))

        if stitched_pts.size > 0:
            robot_pos = np.array([self.x, self.y, self.z], dtype=np.float32)
            dx = stitched_pts[:, 0] - robot_pos[0]
            dy = stitched_pts[:, 1] - robot_pos[1]
            mask = (dx * dx + dy * dy) <= (self.max_stitch_radius ** 2)
            self.stitched_cloud.points = o3d.utility.Vector3dVector(stitched_pts[mask])

        # Publish clean cloud
        if clean_array.shape[0] > 0:
            header           = Header()
            header.stamp     = msg.header.stamp
            header.frame_id  = "odom"
            self.pc_pub.publish(pc2.create_cloud_xyz32(header, clean_array.tolist()))

        # Publish stitched cloud
        stitched_array = np.asarray(self.stitched_cloud.points)
        if stitched_array.shape[0] > 0:
            s_header          = Header()
            s_header.stamp    = msg.header.stamp
            s_header.frame_id = "odom"
            self.stitched_pub.publish(pc2.create_cloud_xyz32(s_header, stitched_array.tolist()))

            # Wall extraction: slice by height band relative to camera
            camera_height = self.z + self.cam_translation[2]
            z_min = camera_height + 0.1
            z_max = camera_height + 0.2
            wall_pts = stitched_array[
                (stitched_array[:, 2] > z_min) & (stitched_array[:, 2] < z_max)
            ]
            if wall_pts.shape[0] > 0:
                w_header          = Header()
                w_header.stamp    = msg.header.stamp
                w_header.frame_id = "odom"
                self.wall_pub.publish(pc2.create_cloud_xyz32(w_header, wall_pts.tolist()))


def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()