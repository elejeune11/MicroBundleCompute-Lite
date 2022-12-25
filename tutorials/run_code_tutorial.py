import argparse
import matplotlib.pyplot as plt
from microbundlecomputelite import create_tissue_mask as ctm
from microbundlecomputelite import image_analysis as ia
from microbundlecomputelite import strain_analysis as sa
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", help="the user input folder location")
args = parser.parse_args()
input_folder_str = args.input_folder

# input_folder_str = "files/example_data"

self_path_file = Path(__file__)
self_path = self_path_file.resolve().parent
input_folder = self_path.joinpath(input_folder_str).resolve()

# automatically create a tissue mask
# (a manual mask can also be used -- just name it "tissue_mask.txt" -- 1=tissue, 0=background)
seg_fcn_num = 1
fname = "tissue_mask"
frame_num = 0
ctm.run_create_tissue_mask(input_folder, seg_fcn_num, fname, frame_num)

# run the tracking
ia.run_tracking(input_folder)

# run the tracking visualization
col_max = 3
col_map = plt.cm.viridis
png_path_list, gif_path = ia.run_visualization(input_folder, col_max, col_map)

# rotate and interpolate tracking results
# rotate the results
input_mask = True  # this will use the mask to determine the rotation vector.
ia.run_rotation(input_folder, input_mask)

# interpolate results
row_vec = np.linspace(215, 305, 12)
col_vec = np.linspace(120, 400, 30)
row_grid, col_grid = np.meshgrid(row_vec, col_vec)
row_sample = row_grid.reshape((-1, 1))
col_sample = col_grid.reshape((-1, 1))
row_col_sample = np.hstack((row_sample, col_sample))
fname = "interpolated_rotated"
ia.run_interpolate(input_folder, row_col_sample, fname, is_rotated=True)

# visualize interpolated tracking results
ia.visualize_interpolate(input_folder, col_max=col_max, col_map=col_map, is_rotated=True, interpolation_fname=fname)

# run the strain analysis (will automatically rotate based on the mask)
pillar_clip_fraction = 0.5
shrink_row = 0.1
shrink_col = 0.1
tile_dim_pix = 40
num_tile_row = 5
num_tile_col = 3
tile_style = 1
sa.run_sub_domain_strain_analysis(input_folder, pillar_clip_fraction, shrink_row, shrink_col, tile_dim_pix, num_tile_row, num_tile_col, tile_style)

# visualize the strain analysis results
col_min = -0.025
col_max = 0.025
col_map = plt.cm.RdBu
sa.visualize_sub_domain_strain(input_folder, col_min, col_max, col_map)
