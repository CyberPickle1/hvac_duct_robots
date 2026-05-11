import heapq
import numpy as np
import trimesh
import matplotlib.pyplot as plt 
from skimage.morphology import skeletonize
import math


# Defining the cell is used to store data and info about said cell
class Cell:
    def __init__(self):
        self.parent = None # Stores floor,row,collum in tuple form
        self.f = float('inf') # total cost of path 
        self.g = float('inf') # cost of the cell (accumlated cost)
        self.h = 0 # cost to goal 

# Main class this needs floors and vertical_connectors to be specified to launch 
class AStarMultiFloor:
    def __init__(self, floors, vertical_connectors, distance_maps=None, skeleton_maps=None):
        self.floors = floors
        self.vertical_connectors = vertical_connectors
        self.num_floors = len(floors)
        self.rows, self.cols = floors[0].shape # extracts the rows and collums of the first floor assuming all floors have the same shape
        self.skeleton_maps = skeleton_maps if skeleton_maps is not None else [None] * self.num_floors

    def is_valid(self, floor, row, col):
        return 0 <= floor < self.num_floors and 0 <= row < self.rows and 0 <= col < self.cols
    
    def is_unblocked(self, floor, row, col):
        return self.floors[floor][row, col] == 1

    def calculate_h_value(self, floor, row, col, dest):
        return ((floor - dest[0])**2 + (row - dest[1])**2 + (col - dest[2])**2) ** 0.5

    def trace_path(self, cell_details, dest):
        path = []
        floor, row, col = dest
        while True:
            path.append((floor, row, col))
            parent = cell_details[floor][row][col].parent
            if parent is None or parent == (floor, row, col):
                break
            floor, row, col = parent
        path.reverse()
        return path

    def a_star_search(self, src, dest):
        if not self.is_valid(*src) or not self.is_valid(*dest):
            return []
        if not self.is_unblocked(*src) or not self.is_unblocked(*dest):
            return []
        cell_details = [[[Cell() for _ in range(self.cols)] for _ in range(self.rows)] for _ in range(self.num_floors)]
        closed_list = [[[False for _ in range(self.cols)] for _ in range(self.rows)] for _ in range(self.num_floors)]

        f_s, r_s, c_s = src
        cell_details[f_s][r_s][c_s].f = 0
        cell_details[f_s][r_s][c_s].g = 0
        cell_details[f_s][r_s][c_s].h = 0
        cell_details[f_s][r_s][c_s].parent = (f_s, r_s, c_s)
        open_list = [] 
        heapq.heappush(open_list, (0.0, src))
        directions_2d = [(0,1),(0,-1),(1,0),(-1,0)]

        while open_list:
            _, (floor, row, col) = heapq.heappop(open_list)
            if closed_list[floor][row][col]:
                continue
            closed_list[floor][row][col] = True

            if (floor, row, col) == dest:
                return self.trace_path(cell_details, dest)

            for dr, dc in directions_2d:
                nr, nc = row + dr, col + dc
                nf = floor
                if self.is_valid(nf, nr, nc) and self.is_unblocked(nf, nr, nc) and not closed_list[nf][nr][nc]:
                    skeleton_bonus = 0
                    if self.skeleton_maps[floor] is not None and self.skeleton_maps[floor][nr, nc]:
                        skeleton_bonus = -0.5  # Negative = reward
                    else:
                        skeleton_bonus = 0.5  # Encourage to move away from wall
                    step_cost = np.hypot(dr, dc) 
                    g_new = cell_details[floor][row][col].g + step_cost + skeleton_bonus
                    h_new = self.calculate_h_value(nf, nr, nc, dest)
                    f_new = g_new + h_new
                    if cell_details[nf][nr][nc].f > f_new:
                        cell_details[nf][nr][nc].f = f_new
                        cell_details[nf][nr][nc].g = g_new
                        cell_details[nf][nr][nc].h = h_new
                        cell_details[nf][nr][nc].parent = (floor, row, col)
                        heapq.heappush(open_list, (f_new, (nf, nr, nc)))

            # Vertical up
            if (row, col) in self.vertical_connectors.get(floor, set()):
                nf = floor + 1
                if nf < self.num_floors and self.is_unblocked(nf, row, col) and not closed_list[nf][row][col]:
                    g_new = cell_details[floor][row][col].g + 2
                    h_new = self.calculate_h_value(nf, row, col, dest)
                    f_new = g_new + h_new
                    if cell_details[nf][row][col].f > f_new:
                        cell_details[nf][row][col].f = f_new
                        cell_details[nf][row][col].g = g_new
                        cell_details[nf][row][col].h = h_new
                        cell_details[nf][row][col].parent = (floor, row, col)
                        heapq.heappush(open_list, (f_new, (nf, row, col)))

            # Vertical down
            if floor > 0 and (row, col) in self.vertical_connectors.get(floor - 1, set()):
                nf = floor - 1
                if self.is_unblocked(nf, row, col) and not closed_list[nf][row][col]:
                    g_new = cell_details[floor][row][col].g + 2
                    h_new = self.calculate_h_value(nf, row, col, dest)
                    f_new = g_new + h_new
                    if cell_details[nf][row][col].f > f_new:
                        cell_details[nf][row][col].f = f_new
                        cell_details[nf][row][col].g = g_new
                        cell_details[nf][row][col].h = h_new
                        cell_details[nf][row][col].parent = (floor, row, col)
                        heapq.heappush(open_list, (f_new, (nf, row, col)))

        return []

def multi_floor_multi_target_astar(floors, vertical_connectors, start_floor, start_pos, floor_targets, astar):
    total_path = []
    current_floor = start_floor
    current_pos = start_pos
    num_floors = len(floors)

    for floor_idx in range(current_floor, num_floors):
        targets = floor_targets.get(floor_idx, []).copy()
        while targets:
            best_t, best_path, best_len = None, None, float('inf')
            for t in targets:
                p = astar.a_star_search((floor_idx, *current_pos), (floor_idx, *t))
                if p and len(p) < best_len:
                    best_t, best_path, best_len = t, p, len(p)
            if best_path is None:          # nothing reachable
                break
            total_path.extend(best_path[1:] if total_path else best_path)
            current_pos = best_t
            targets.remove(best_t)

        # Move to vertical connector for next floor if any
        if floor_idx + 1 < num_floors:
            connectors = vertical_connectors.get(floor_idx, set())
            if not connectors:
                break
            next_connector = min(connectors, key=lambda c: (c[0]-current_pos[0])**2 + (c[1]-current_pos[1])**2)
            connector_path = astar.a_star_search((floor_idx, *current_pos), (floor_idx, *next_connector))
            if not connector_path:
                print(f"No path to vertical connector {next_connector} on floor {floor_idx}, stopping.")
                break
            if total_path:
                connector_path = connector_path[1:]
            total_path.extend(connector_path)
            total_path.append((floor_idx+1, *next_connector))  # vertical move
            current_pos = next_connector
        else:
            break
    return total_path

def downsample_path(path, step=5):
    if len(path) == 0:
        return path
    downsampled = path[::step]
    if path[-1] != downsampled[-1]:
        downsampled.append(path[-1])
    return downsampled

def occupnacy_grid(pitch):
    mesh_path = r"\\wsl$\Ubuntu-22.04\home\josephdoingjosephthings\crawler\src\sdf\cleanduct3\cleanduct3.stl"

    relative_height_threshold = 0.5  # m
    base_height = 3.0               # m

    mesh = trimesh.load(mesh_path)

    # Check if mesh is valid
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # X‑Y bounds
    bounds_min = mesh.bounds[0][:2]   # (x_min, y_min)
    bounds_max = mesh.bounds[1][:2]   # (x_max, y_max)

    # Grid size (cols = X, rows = Y)
    grid_cols = int(np.ceil((bounds_max[0] - bounds_min[0]) / pitch))
    grid_rows = int(np.ceil((bounds_max[1] - bounds_min[1]) / pitch))

    # Occ[row, col] where row→Y, col→X
    occupancy_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    vox = mesh.voxelized(pitch)
    inds = np.argwhere(vox.matrix)
    pts = vox.indices_to_points(inds)  # XYZ voxel centres

    # World → grid indices
    xy = pts[:, :2]
    z = pts[:, 2]

    ij = ((xy - bounds_min) / pitch).astype(int)   # (col, row)
    cols = ij[:, 0]
    rows = ij[:, 1]

    for row, col, zc in zip(rows, cols, z):
        if 0 <= row < grid_rows and 0 <= col < grid_cols:
            if abs(zc - base_height) > relative_height_threshold:
                occupancy_grid[row, col] = 2  # tall
            else:
                if occupancy_grid[row, col] != 2:
                    occupancy_grid[row, col] = 1  # normal

    # Real‑world extents for imshow
    extent = [
        bounds_min[0],
        bounds_min[0] + grid_cols * pitch,
        bounds_min[1],
        bounds_min[1] + grid_rows * pitch
    ]

    plt.figure(figsize=(8, 8))
    plt.imshow(occupancy_grid, origin='lower', extent=extent,
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='0 free   1 normal   2 tall')
    plt.xlabel('X ')
    plt.ylabel('Y ')
    plt.title('Occupancy Grid')
    plt.tight_layout()
    plt.show()

    occupancy_grid = occupancy_grid.T 
    print(f"Grid1 shape (rows, cols): {occupancy_grid.shape}")



    return occupancy_grid, bounds_min , extent


def indices_to_meters(path, bounds_min, pitch=0.05):
    x_min, y_min = bounds_min

    return [(floor,
             x_min + row * pitch,  # X
             y_min + col * pitch)  # Y
            for (floor, row, col) in path]



if __name__ == "__main__":
    
    grid1, bounds_min , extent  = occupnacy_grid(0.05)


    grid2 = [
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 1] 
    ]
    # Binary occupancy: 1 = free, 0 = wall or obstacle
    binary_free = (grid1 == 1)

    # Skeletonize free space
    skeleton = skeletonize(binary_free)

    # Optional: visualize
    plt.imshow(skeleton.T, origin='lower', cmap='gray')
    plt.title("Skeleton of Duct Interior")
    plt.show()

    floors = [np.array(grid1), np.array(grid2)]
    vertical_connectors = {
        0: {(8, 8)},
        1: {(0, 3)}
    }

    start = (0, 525, 200)
    floor_targets = {
                0: [
                        (349, 188)#, (448, 190), (449, 29),
                        # (348, 10),  (72, 30), #(227, 3), #problamatic 
                        # (18, 28), (110, 230), (20, 235),
                        # (22, 108), (47, 248), (427, 244),
                        # (427, 431), (197, 431), (170, 437),
                        # (39, 437)

                ]
            }

    start_floor = start[0]
    start_pos = (start[1], start[2])
    
    astar = AStarMultiFloor(
    floors,
    vertical_connectors,
    skeleton_maps=[skeleton] + [None] * (len(floors) - 1)
)

    path = multi_floor_multi_target_astar(
        floors,
        vertical_connectors,
        start_floor,
        start_pos,
        floor_targets,
        astar
    )
    print(f"Grid1 shape (rows, cols): {grid1.shape}")




    # print("Computed path:")
    # for p in path:
    #     print(p)

        #os.makedirs("exported", exist_ok=True)
# # === Export raw path (indices) as a Python file ===
#     output_file = r"\\wsl.localhost\Ubuntu-22.04\home\josephdoingjosephthings\crawler\src\pypackage\pypackage\precomputed_path.py"

#     with open(output_file, "w") as f:
#         f.write("precomputed_path = [\n")
#         for point in path:
#             f.write(f"    {point},\n")
#         f.write("]\n")

#     print(f"Saved path (indices) to {output_file}")


    path_in_meters = indices_to_meters(path, bounds_min , pitch=0.05)



    # print("Path in meters:")
    # for p in path_in_meters:
    #     print(p)



    # Check if all targets were reached
    reached_targets = set(p for p in path if p[0] == 0 and (p[1], p[2]) in floor_targets[0])
    missing_targets = set(floor_targets[0]) - set((t[1], t[2]) for t in reached_targets)
    if missing_targets:
        print(f"Warning: Could not reach targets {missing_targets} on floor 0")
    else:
        print("All targets reached on floor 0!")
        plt.figure(figsize=(8, 8))
    plt.imshow(grid1.T, origin='lower', extent=extent,
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='0 free   1 normal   2 tall')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Occupancy Grid with Planned Path')

    xs = [p[1] for p in path_in_meters if p[0] == 0]
    ys = [p[2] for p in path_in_meters if p[0] == 0]

    plt.plot(xs, ys, color='red', linewidth=1, marker='.', markersize=1.5, label='Planned Path')

    plt.legend()
    plt.tight_layout()
    plt.show()

        # Plot downsampled path
    downsampled = downsample_path(path, step=5)
    downsampled_meters = indices_to_meters(downsampled, bounds_min, pitch=0.05)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid1.T, origin='lower', extent=extent,
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='0 free   1 normal   2 tall')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Downsampled Path Over Occupancy Grid')

    ds_xs = [p[1] for p in downsampled_meters if p[0] == 0]
    ds_ys = [p[2] for p in downsampled_meters if p[0] == 0]

    plt.plot(ds_xs, ds_ys, color='orange', linewidth=1, marker='.', markersize=1.5, label='Downsampled Path')

    plt.legend()
    plt.tight_layout()
    plt.show()




    def clean_point_cloud(self, cloud):
        if len(cloud.points) == 0:
            return cloud  # for initial empty point cloud

        # Remove statistical outliers
        cl, ind = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        cleaned = cloud.select_by_index(ind)

        # Voxel downsampling
        downsampled = cleaned.voxel_down_sample(voxel_size=0.05) 

        return downsampled
    
    def compose_transforms(self, robot_pos, robot_rot_quat, cam_translation, cam_rot_quat):
        
        # Rotation matrices
        r_robot = R.from_quat(robot_rot_quat)
        r_cam = R.from_quat(cam_rot_quat)
        
        # Compose rotation
        r_total = r_robot * r_cam
        
        # Compose translation: robot translation + rotated camera offset
        t_total = robot_pos + r_robot.apply(cam_translation)
        
        # Return composed translation and quaternion
        return t_total, r_total.as_quat()
    
    def path_predictor(self, index, path, point_cloud, steps=1, linear_tolerance=0.1, angular_tolerance=math.radians(5), step_resolution=0.05):
        if point_cloud is None or len(point_cloud.points) == 0:
            self.get_logger().warn("[PREDICT] No usable point cloud!")
            return False

        pc_np = np.asarray(point_cloud.points)

        # Broader Z-filtering to accommodate camera tilt and surface unevenness
        z_min = self.z - 0.05
        z_max = self.z + 0.10
        pc_filtered = pc_np[(pc_np[:, 2] > z_min) & (pc_np[:, 2] < z_max)]

        if len(pc_filtered) == 0:
            self.get_logger().warn("[PREDICT] Point cloud has no relevant points after Z-filtering!")
            return False

        pc_xy = pc_filtered[:, :2]

        # Define inflated robot footprint
        robot_width = 0.13
        robot_length = 0.1
        padding = 0.05  # Safety buffer
        half_w = (robot_width + padding) / 2
        half_l = (robot_length + padding) / 2

        base_footprint = np.array([
            [ half_l,  half_w],
            [ half_l, -half_w],
            [-half_l, -half_w],
            [-half_l,  half_w]
        ])

        # Start with current pose
        x_robot = self.x + self.odom_offset_x
        y_robot = self.y + self.odom_offset_y
        theta_robot = self.theta_current

        for i in range(index, min(index + steps, len(path) - 1)):
            _, x0, y0 = path[i]
            _, x1, y1 = path[i + 1]

            dx = x1 - x0
            dy = y1 - y0
            segment_dist = math.hypot(dx, dy)
            heading = math.atan2(dy, dx)

            angle_diff = heading - theta_robot
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            aligned = abs(angle_diff) <= angular_tolerance

            if not aligned:
                self.get_logger().info(f"[PREDICT] Skipping prediction at index {i} — misaligned (angle_diff: {math.degrees(angle_diff):.2f}°)")
                return False

            num_steps = max(1, int(segment_dist / step_resolution))

            for s in range(num_steps):
                ratio = (s + 1) / num_steps
                interp_x = x0 + ratio * dx
                interp_y = y0 + ratio * dy
                interp_theta = heading  # assumed constant heading for short segments

                # Transform robot footprint at predicted pose
                rot = np.array([
                    [math.cos(interp_theta), -math.sin(interp_theta)],
                    [math.sin(interp_theta),  math.cos(interp_theta)]
                ])
                transformed = (rot @ base_footprint.T).T + np.array([interp_x, interp_y])
                poly = Path(transformed)

                inside = poly.contains_points(pc_xy)
                if np.any(inside):
                    self.get_logger().warn(
                        f"[PREDICT] Collision predicted at segment {i}, step {s}, pose=({interp_x:.2f}, {interp_y:.2f}), heading={math.degrees(interp_theta):.1f}°"
                    )
                    return True

            # Update pose estimate for chained prediction
            x_robot = x1
            y_robot = y1
            theta_robot = heading

        return False