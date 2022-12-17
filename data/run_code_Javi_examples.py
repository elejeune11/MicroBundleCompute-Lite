import matplotlib.pyplot as plt
from microbundlecomputelite import image_analysis as ia
import numpy as np
from pathlib import Path


# run the aligned example
input_folder = Path("Javi_data/align0.1LAP_05")

# run the tracking
ia.run_tracking(input_folder)

# run the visualization
col_max = 10
col_map = plt.cm.viridis
png_path_list, gif_path = ia.run_visualization(input_folder, col_max, col_map)

# rotate the results
input_mask = True  # this will use the mask to determine the rotation vector.
ia.run_rotation(input_folder, input_mask)

# visualize the rotated results
ia.run_rotation_visualization(input_folder, col_max, col_map)

# interpolate results
row_vec = np.linspace(325, 325 + 60, 6)
col_vec = np.linspace(160, 575, 30)
row_grid, col_grid = np.meshgrid(row_vec, col_vec)
row_sample = row_grid.reshape((-1, 1))
col_sample = col_grid.reshape((-1, 1))
row_col_sample = np.hstack((row_sample, col_sample))
fname = "interpolated_rotated"
ia.run_interpolate(input_folder, row_col_sample, fname, is_rotated=True)
ia.visualize_interpolate(input_folder, col_max=col_max, col_map=col_map, is_rotated=True, interpolation_fname=fname)

#  translate results (not visualized)
pixel_origin_row = 100  # change these values to what you want
pixel_origin_col = 150  # change these values to what you want
microns_per_pixel_row = 0.25  # change these values to what you want
microns_per_pixel_col = 0.25  # change these values to what you want
use_rotated = True  # change this if you want to use the original
fname = "interpolated_rotated"
new_fname = "interpolated_rotated_scaled_"
saved_paths = ia.run_scale_and_center_coordinates(input_folder, pixel_origin_row, pixel_origin_col, microns_per_pixel_row, microns_per_pixel_col, use_rotated, fname, new_fname)

###############################################################################
###############################################################################
###############################################################################

# run the random example
input_folder = Path("Javi_data/rand5.0LAP_softPosts_03")

# run the tracking
ia.run_tracking(input_folder)

# run the visualization
col_max = 3
col_map = plt.cm.viridis
png_path_list, gif_path = ia.run_visualization(input_folder, col_max, col_map)

# rotate the results
input_mask = True  # this will use the mask to determine the rotation vector.
ia.run_rotation(input_folder, input_mask)

# visualize the rotated results
ia.run_rotation_visualization(input_folder, col_max, col_map)

# interpolate results
row_vec = np.linspace(340, 340 + 60, 6)
col_vec = np.linspace(160, 575, 30)
row_grid, col_grid = np.meshgrid(row_vec, col_vec)
row_sample = row_grid.reshape((-1, 1))
col_sample = col_grid.reshape((-1, 1))
row_col_sample = np.hstack((row_sample, col_sample))
fname = "interpolated_rotated"
ia.run_interpolate(input_folder, row_col_sample, fname, is_rotated=True)
ia.visualize_interpolate(input_folder, col_max=col_max, col_map=col_map, is_rotated=True, interpolation_fname=fname)

#  translate results (not visualized)
pixel_origin_row = 100  # change these values to what you want
pixel_origin_col = 150  # change these values to what you want
microns_per_pixel_row = 0.25  # change these values to what you want
microns_per_pixel_col = 0.25  # change these values to what you want
use_rotated = True  # change this if you want to use the original
fname = "interpolated_rotated"
new_fname = "interpolated_rotated_scaled_"
saved_paths = ia.run_scale_and_center_coordinates(input_folder, pixel_origin_row, pixel_origin_col, microns_per_pixel_row, microns_per_pixel_col, use_rotated, fname, new_fname)
