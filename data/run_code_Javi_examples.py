import matplotlib.pyplot as plt
from microbundlecomputelite import create_tissue_mask as ctm
from microbundlecomputelite import image_analysis as ia
from microbundlecomputelite import strain_analysis as sa
import numpy as np
from pathlib import Path

###############################################################################
###############################################################################
# ALIGNED EXAMPLE
###############################################################################
###############################################################################

# run the aligned example
self_path_file = Path(__file__)
self_path = self_path_file.resolve().parent
input_folder = self_path.joinpath("Javi_data").resolve().joinpath("align0.1LAP_05").resolve()

# automatically create the tissue mask
seg_fcn_num = 2
fname = "tissue_mask"
frame_num = 0
ctm.run_create_tissue_mask(input_folder, seg_fcn_num, fname, frame_num)

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
col_vec = np.linspace(125, 540, 30)
row_grid, col_grid = np.meshgrid(row_vec, col_vec)
row_sample = row_grid.reshape((-1, 1))
col_sample = col_grid.reshape((-1, 1))
row_col_sample = np.hstack((row_sample, col_sample))
fname = "interpolated_rotated"
ia.run_interpolate(input_folder, row_col_sample, fname, is_rotated=True)
ia.visualize_interpolate(input_folder, col_max=col_max, col_map=col_map, is_rotated=True, interpolation_fname=fname)

# translate results (not visualized)
pixel_origin_row = 100  # change these values to what you want
pixel_origin_col = 150  # change these values to what you want
microns_per_pixel_row = 0.25  # change these values to what you want
microns_per_pixel_col = 0.25  # change these values to what you want
use_rotated = True  # change this if you want to use the original
fname = "interpolated_rotated"
new_fname = "interpolated_rotated_scaled_"
saved_paths = ia.run_scale_and_center_coordinates(input_folder, pixel_origin_row, pixel_origin_col, microns_per_pixel_row, microns_per_pixel_col, use_rotated, fname, new_fname)

# compute and visualize strain
ia.run_tracking(input_folder)
pillar_clip_fraction = 0.5
shrink_row = 0.1
shrink_col = 0.1
tile_dim_pix = 40
num_tile_row = 5
num_tile_col = 3
tile_style = 1
sa.run_sub_domain_strain_analysis(input_folder, pillar_clip_fraction, shrink_row, shrink_col, tile_dim_pix, num_tile_row, num_tile_col, tile_style)
col_min = -0.1
col_max = 0.1
col_map = plt.cm.RdBu
sa.visualize_sub_domain_strain(input_folder, col_min, col_max, col_map)

###############################################################################
###############################################################################
# RANDOM EXAMPLE
###############################################################################
###############################################################################

# run the random example
self_path_file = Path(__file__)
self_path = self_path_file.resolve().parent
input_folder = self_path.joinpath("Javi_data").resolve().joinpath("rand5.0LAP_softPosts_03").resolve()

# automatically create the tissue mask
seg_fcn_num = 2
fname = "tissue_mask"
frame_num = 0
ctm.run_create_tissue_mask(input_folder, seg_fcn_num, fname, frame_num)

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
col_vec = np.linspace(170, 585, 30)
row_grid, col_grid = np.meshgrid(row_vec, col_vec)
row_sample = row_grid.reshape((-1, 1))
col_sample = col_grid.reshape((-1, 1))
row_col_sample = np.hstack((row_sample, col_sample))
fname = "interpolated_rotated"
ia.run_interpolate(input_folder, row_col_sample, fname, is_rotated=True)
ia.visualize_interpolate(input_folder, col_max=col_max, col_map=col_map, is_rotated=True, interpolation_fname=fname)

# translate results (not visualized)
pixel_origin_row = 100  # change these values to what you want
pixel_origin_col = 150  # change these values to what you want
microns_per_pixel_row = 0.25  # change these values to what you want
microns_per_pixel_col = 0.25  # change these values to what you want
use_rotated = True  # change this if you want to use the original
fname = "interpolated_rotated"
new_fname = "interpolated_rotated_scaled_"
saved_paths = ia.run_scale_and_center_coordinates(input_folder, pixel_origin_row, pixel_origin_col, microns_per_pixel_row, microns_per_pixel_col, use_rotated, fname, new_fname)

# compute and visualize strain
ia.run_tracking(input_folder)
pillar_clip_fraction = 0.5
shrink_row = 0.1
shrink_col = 0.1
tile_dim_pix = 40
num_tile_row = 5
num_tile_col = 3
tile_style = 1
sa.run_sub_domain_strain_analysis(input_folder, pillar_clip_fraction, shrink_row, shrink_col, tile_dim_pix, num_tile_row, num_tile_col, tile_style)
col_min = -0.025
col_max = 0.025
col_map = plt.cm.RdBu
sa.visualize_sub_domain_strain(input_folder, col_min, col_max, col_map)
