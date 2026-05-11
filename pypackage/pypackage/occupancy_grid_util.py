import trimesh
import numpy as np


def occupnacy_grid(pitch):
    mesh_path = ("/home/josephdoingjosephthings/crawler/src/sdf/cleanduct3/cleanduct3.stl")
    relative_height_threshold = 0.5  # m
    base_height = 3.0               # m

    mesh = trimesh.load(mesh_path)
    # X‑Y bounds
    bounds_min = mesh.bounds[0][:2]   # (x_min, y_min)
    bounds_max = mesh.bounds[1][:2]   # (x_max, y_max)

    # Grid size
    grid_cols = int(np.ceil((bounds_max[0] - bounds_min[0]) / pitch))
    grid_rows = int(np.ceil((bounds_max[1] - bounds_min[1]) / pitch))

    # Occ[row, col]
    occupancy_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    vox = mesh.voxelized(pitch)
    inds = np.argwhere(vox.matrix)
    pts = vox.indices_to_points(inds)    

    # World to grid 
    xy = pts[:, :2]
    z  = pts[:, 2]
    ij = ((xy - bounds_min) / pitch).astype(int)   # (col, row)
    cols = ij[:, 0]
    rows = ij[:, 1]

    for r, c, zc in zip(rows, cols, z):
        if 0 <= r < grid_rows and 0 <= c < grid_cols:
            if abs(zc - base_height) > relative_height_threshold:
                occupancy_grid[r, c] = 2      # tall
            else:
                if occupancy_grid[r, c] != 2:
                    occupancy_grid[r, c] = 1  # normal

    # Real‑world extents for imshow
    extent = [
        bounds_min[0],
        bounds_min[0] + grid_cols * pitch,
        bounds_min[1],
        bounds_min[1] + grid_rows * pitch
    ]

    # plt.figure(figsize=(8, 8))
    # plt.imshow(occupancy_grid, origin='lower', extent=extent,
    #            cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='0 free   1 normal   2 tall')
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')
    # plt.title('Occupancy Grid (metres)')
    # plt.tight_layout()
    # plt.show()
    occupancy_grid = occupancy_grid.T
    return occupancy_grid , bounds_min
