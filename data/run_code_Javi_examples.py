import matplotlib.pyplot as plt
from microbundlecomputelite import image_analysis as ia
from pathlib import Path



# run the aligned example
input_folder = Path("Javi_data/align0.1LAP_05")

# run the tracking
ia.run_tracking(input_folder)

# run the visualization
col_max = 10
col_map = plt.cm.viridis
png_path_list, gif_path, mp4_path = ia.run_visualization(input_folder, col_max, col_map)


# run the random example
input_folder = Path("Javi_data/rand5.0LAP_softPosts_03")

# run the tracking
ia.run_tracking(input_folder)

# run the visualization
col_max = 3
col_map = plt.cm.viridis
png_path_list, gif_path, mp4_path = ia.run_visualization(input_folder, col_max, col_map)


