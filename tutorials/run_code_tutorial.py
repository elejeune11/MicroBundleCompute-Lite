import argparse
import matplotlib.pyplot as plt
from microbundlecomputelite import image_analysis as ia
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", help="the user input folder location")
args = parser.parse_args()
input_folder_str = args.input_folder
input_folder = Path(input_folder_str)

# run the tracking
ia.run_tracking(input_folder)

# run the visualization
col_max = 3
col_map = plt.cm.viridis
png_path_list, gif_path = ia.run_visualization(input_folder, col_max, col_map)
