import cv2
import glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
# import moviepy.editor as mp
import numpy as np
import os
from pathlib import Path
from skimage import measure
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.signal import find_peaks
from skimage import exposure
from skimage import img_as_ubyte
from skimage import io
from typing import List, Union


def hello_microbundle_compute() -> str:
    "Given no input. Simple hello world as a test function."
    return "Hello World!"


def read_tiff(img_path: Path) -> np.ndarray:
    """Given a path to a tiff. Will return an array."""
    img = io.imread(img_path)
    return img


def image_folder_to_path_list(folder_path: Path) -> List:
    """Given a folder path. Will return the path to all files in that path in order."""
    name_list = glob.glob(str(folder_path) + '/*.TIF')
    name_list.sort()
    name_list_path = []
    for name in name_list:
        name_list_path.append(Path(name))
    return name_list_path


def read_all_tiff(path_list: List) -> List:
    """Given a folder path. Will return a list of all tiffs as an array."""
    tiff_list = []
    for path in path_list:
        array = read_tiff(path)
        tiff_list.append(array)
    return tiff_list


def create_folder(folder_path: Path, new_folder_name: str) -> Path:
    """Given a path to a directory and a folder name. Will create a directory in the given directory."""
    new_path = folder_path.joinpath(new_folder_name).resolve()
    if new_path.exists() is False:
        os.mkdir(new_path)
    return new_path


def uint16_to_uint8(img_16: np.ndarray) -> np.ndarray:
    """Given a uint16 image. Will normalize + rescale and convert to uint8."""
    img_8 = img_as_ubyte(exposure.rescale_intensity(img_16))
    return img_8


def bool_to_uint8(arr_bool: np.ndarray) -> np.ndarray:
    """Given a boolean array. Will return a uint8 array."""
    arr_uint8 = (1. * arr_bool).astype('uint8')
    return arr_uint8


def uint16_to_uint8_all(img_list: List) -> List:
    """Given an image list of uint16. Will return the same list all as uint8."""
    uint8_list = []
    for img in img_list:
        img8 = uint16_to_uint8(img)
        uint8_list.append(img8)
    return uint8_list


def read_txt_as_mask(file_path: Path) -> np.ndarray:
    """Given a path to a saved txt file array. Will return an array formatted as unit8."""
    img = np.loadtxt(file_path)
    img_uint8 = bool_to_uint8(img)
    return img_uint8


def get_tracking_param_dicts() -> dict:
    """Will return dictionaries specifying the feature parameters and tracking parameters.
    In future, these may vary based on version."""
    feature_params = dict(maxCorners=10000, qualityLevel=0.1, minDistance=5, blockSize=5)
    window = 10
    lk_params = dict(winSize=(window, window), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    return feature_params, lk_params


def mask_to_track_points(img_uint8: np.ndarray, mask: np.ndarray, feature_params: dict) -> np.ndarray:
    """Given an image and a mask. Will return the good features to track within the mask region."""
    # ensure that the mask is uint8
    mask_uint8 = bool_to_uint8(mask)
    track_points_0 = cv2.goodFeaturesToTrack(img_uint8, mask=mask_uint8, **feature_params)
    return track_points_0


def track_one_step(img_uint8_0: np.ndarray, img_uint8_1: np.ndarray, track_points_0: np.ndarray, lk_params: dict):
    """Given img_0, img_1, tracking points p0, and tracking parameters.
    Will return the tracking points new location. Note that for now standard deviation and error are ignored."""
    track_points_1, _, _ = cv2.calcOpticalFlowPyrLK(img_uint8_0, img_uint8_1, track_points_0, None, **lk_params)
    return track_points_1


def track_all_steps(img_list_uint8: List, mask: np.ndarray) -> np.ndarray:
    """Given the image list, mask, and order. Will run tracking through the whole img list in order.
    Note that the returned order of tracked points will match order_list."""
    feature_params, lk_params = get_tracking_param_dicts()
    img_0 = img_list_uint8[0]
    track_points = mask_to_track_points(img_0, mask, feature_params)
    num_track_pts = track_points.shape[0]
    num_imgs = len(img_list_uint8)
    tracker_0 = np.zeros((num_track_pts, num_imgs))
    tracker_1 = np.zeros((num_track_pts, num_imgs))
    for kk in range(0, num_imgs - 1):
        tracker_0[:, kk] = track_points[:, 0, 0]
        tracker_1[:, kk] = track_points[:, 0, 1]
        img_0 = img_list_uint8[kk]
        img_1 = img_list_uint8[kk + 1]
        track_points = track_one_step(img_0, img_1, track_points, lk_params)
    tracker_0[:, kk + 1] = track_points[:, 0, 0]
    tracker_1[:, kk + 1] = track_points[:, 0, 1]
    return tracker_0, tracker_1


def compute_abs_position_timeseries(tracker_0: np.ndarray, tracker_1: np.ndarray) -> np.ndarray:
    """Given tracker arrays. Will return single timeseries of absolute displacement."""
    disp_0_all = np.zeros(tracker_0.shape)
    disp_1_all = np.zeros(tracker_1.shape)
    for kk in range(tracker_0.shape[1]):
        disp_0_all[:, kk] = tracker_0[:, kk] - tracker_0[:, 0]
        disp_1_all[:, kk] = tracker_1[:, kk] - tracker_1[:, 0]
    disp_abs_all = (disp_0_all ** 2.0 + disp_1_all ** 2.0) ** 0.5
    disp_abs_mean = np.mean(disp_abs_all, axis=0)
    disp_abs_all = ((disp_0_all) ** 2.0 + (disp_1_all) ** 2.0) ** 0.5
    return disp_abs_mean, disp_abs_all


def get_time_segment_param_dicts() -> dict:
    """Will return dictionaries specifying the parameters for timeseries segmentation.
    In future, these may vary based on version and/or be computed automatically (e.g., look at spectral info)."""
    time_seg_params = dict(peakDist=20)
    return time_seg_params


def compute_valleys(timeseries: np.ndarray) -> np.ndarray:
    """Given a timeseries. Will compute peaks and valleys."""
    time_seg_params = get_time_segment_param_dicts()
    peaks, _ = find_peaks(timeseries, distance=time_seg_params["peakDist"])
    valleys = []
    for kk in range(0, len(peaks) - 1):
        valleys.append(int(0.5 * peaks[kk] + 0.5 * peaks[kk + 1]))
    info = []
    for kk in range(0, len(valleys) - 1):
        # beat number, start index wrt movie, end index wrt movie
        info.append([kk, valleys[kk], valleys[kk + 1]])
    return np.asarray(info)


def split_tracking(tracker_0: np.ndarray, tracker_1: np.ndarray, info: np.ndarray) -> Path:
    """Given full tracking arrays and info. Will split tracking array by beat according to info."""
    tracker_0_all = []
    tracker_1_all = []
    for kk in range(0, info.shape[0]):
        ix_start = info[kk, 1]
        ix_end = info[kk, 2]
        tracker_0_all.append(tracker_0[:, ix_start:ix_end])
        tracker_1_all.append(tracker_1[:, ix_start:ix_end])
    return tracker_0_all, tracker_1_all


def save_tracking(folder_path: Path, tracker_0_all: List, tracker_1_all: List, info: np.ndarray) -> List:
    """Given tracking results. Will save as text files."""
    new_path = create_folder(folder_path, "results")
    num_beats = info.shape[0]
    saved_paths = []
    for kk in range(0, num_beats):
        file_path = new_path.joinpath("beat%i_row.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), tracker_1_all[kk])
        file_path = new_path.joinpath("beat%i_col.txt" % (kk)).resolve()
        np.savetxt(str(file_path), tracker_0_all[kk])
        saved_paths.append(file_path)
    file_path = new_path.joinpath("info.txt").resolve()
    np.savetxt(str(file_path), info)
    saved_paths.append(file_path)
    return saved_paths


def run_tracking(folder_path: Path) -> List:
    """Given a folder path. Will perform tracking and save results as text files."""
    # read images and mask file
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = image_folder_to_path_list(movie_folder_path)
    tiff_list = read_all_tiff(name_list_path)
    img_list_uint8 = uint16_to_uint8_all(tiff_list)
    mask_file_path = folder_path.joinpath("masks").resolve().joinpath("tissue_mask.txt").resolve()
    mask = read_txt_as_mask(mask_file_path)
    # perform tracking
    tracker_0, tracker_1 = track_all_steps(img_list_uint8, mask)
    # perform timeseries segmentation
    timeseries, _ = compute_abs_position_timeseries(tracker_0, tracker_1)
    info = compute_valleys(timeseries)
    # split tracking results
    tracker_0_all, tracker_1_all = split_tracking(tracker_0, tracker_1, info)
    # save tracking results
    saved_paths = save_tracking(folder_path, tracker_0_all, tracker_1_all, info)
    return saved_paths


def load_tracking_results(folder_path: Path) -> List:
    """Given the folder path. Will load tracking results. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("results").resolve()
    if res_folder_path.exists() is False:
        raise FileNotFoundError("tracking results are not present -- tracking must be run before visualization")
    num_files = len(glob.glob(str(res_folder_path) + "/*.txt"))
    num_beats = int((num_files - 1) / 2)
    tracker_row_all = []
    tracker_col_all = []
    for kk in range(0, num_beats):
        tracker_row = np.loadtxt(str(res_folder_path) + "/beat%i_row.txt" % (kk))
        tracker_col = np.loadtxt(str(res_folder_path) + "/beat%i_col.txt" % (kk))
        tracker_row_all.append(tracker_row)
        tracker_col_all.append(tracker_col)
    info = np.loadtxt(str(res_folder_path) + "/info.txt")
    info_reshape = np.reshape(info, (-1, 3))
    return tracker_row_all, tracker_col_all, info_reshape


def create_pngs(
    folder_path: Path,
    tiff_list: List,
    tracker_row_all: List,
    tracker_col_all: List,
    info: np.ndarray,
    col_max: Union[float, int],
    col_map: object
) -> List:
    """Given tracking results. Will create png version of the visualizations."""
    vis_folder_path = create_folder(folder_path, "visualizations")
    pngs_folder_path = create_folder(vis_folder_path, "pngs")
    path_list = []
    num_beats = info.shape[0]
    for beat in range(0, num_beats):
        tracker_row = tracker_row_all[beat]
        tracker_col = tracker_col_all[beat]
        _, disp_all = compute_abs_position_timeseries(tracker_row, tracker_col)
        start_idx = int(info[beat, 1])
        end_idx = int(info[beat, 2])
        for kk in range(start_idx, end_idx):
            plt.figure()
            plt.imshow(tiff_list[kk], cmap=plt.cm.gray)
            jj = kk - start_idx
            plt.scatter(tracker_col[:, jj], tracker_row[:, jj], c=disp_all[:, jj], s=10, cmap=col_map, vmin=0, vmax=col_max)
            plt.title("frame %i, beat %i" % (kk, beat))
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().labelpad = 15
            cbar.set_label('absolute displacement (pixels)', rotation=270)
            plt.axis("off")
            path = pngs_folder_path.joinpath("%04d_disp.png" % (kk)).resolve()
            plt.savefig(str(path))
            plt.close()
            path_list.append(path)
    return path_list


def create_gif(folder_path: Path, png_path_list: List) -> Path:
    """Given the pngs path list. Will create a gif."""
    img_list = []
    for pa in png_path_list:
        img_list.append(imageio.imread(pa))
    gif_path = folder_path.joinpath("visualizations").resolve().joinpath("abs_disp.gif").resolve()
    imageio.mimsave(str(gif_path), img_list)
    return gif_path


# def create_mp4(folder_path: Path, gif_path: Path) -> Path:
#     """Given the gif path. Will create a mp4."""
#     clip = mp.VideoFileClip(str(gif_path))
#     mp4_path = folder_path.joinpath("visualizations").resolve().joinpath("abs_disp.mp4").resolve()
#     clip.write_videofile(str(mp4_path))
#     return mp4_path


def run_visualization(folder_path: Path, col_max: Union[int, float] = 10, col_map: object = plt.cm.viridis) -> List:
    """Given a folder path where tracking has already been run. Will save visualizations."""
    # read image files
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = image_folder_to_path_list(movie_folder_path)
    tiff_list = read_all_tiff(name_list_path)
    # read tracking results
    tracker_row_all, tracker_col_all, info = load_tracking_results(folder_path)
    # create pngs
    png_path_list = create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, col_max, col_map)
    # create gif
    gif_path = create_gif(folder_path, png_path_list)
    # create mp4
    # mp4_path = create_mp4(folder_path, gif_path)
    return png_path_list, gif_path


def interpolate_points(
    row_col_pos: np.ndarray,
    row_col_vals: np.ndarray,
    row_col_sample: np.ndarray,
) -> np.ndarray:
    """Given row/column positions, row/column values, and sample positions.
    Will interpolate the values and return values at the sample positions."""
    # fit interpolation function and perform interpolation
    row_sample_vals = RBFInterpolator(row_col_pos, row_col_vals[:, 0])(row_col_sample)
    col_sample_vals = RBFInterpolator(row_col_pos, row_col_vals[:, 1])(row_col_sample)
    row_col_sample_vals = np.hstack((row_sample_vals.reshape((-1, 1)), col_sample_vals.reshape((-1, 1))))
    return row_col_sample_vals


def compute_distance(x1: Union[int, float], x2: Union[int, float], y1: Union[int, float], y2: Union[int, float]) -> Union[int, float]:
    """Given two 2D points. Will return the distance between them."""
    dist = ((x1 - x2) ** 2.0 + (y1 - y2) ** 2.0) ** 0.5
    return dist


def compute_unit_vector(x1: Union[int, float], x2: Union[int, float], y1: Union[int, float], y2: Union[int, float]) -> np.ndarray:
    """Given two 2D points. Will return the unit vector between them"""
    dist = compute_distance(x1, x2, y1, y2)
    vec = np.asarray([(x2 - x1) / dist, (y2 - y1) / dist])
    return vec


def insert_borders(mask: np.ndarray, border: int = 10) -> np.ndarray:
    """Given a mask. Will make the borders around it 0."""
    mask[0:border, :] = 0
    mask[-border:, :] = 0
    mask[:, 0:border] = 0
    mask[:, -border:] = 0
    return mask


def box_to_unit_vec(box: np.ndarray) -> np.ndarray:
    """Given the rectangular box. Will compute the unit vector of the longest side."""
    side_1 = compute_distance(box[0, 0], box[1, 0], box[0, 1], box[1, 1])
    side_2 = compute_distance(box[1, 0], box[2, 0], box[1, 1], box[2, 1])
    if side_1 > side_2:
        # side_1 is the long axis
        vec = compute_unit_vector(box[0, 0], box[1, 0], box[0, 1], box[1, 1])
    else:
        # side_2 is the long axis
        vec = compute_unit_vector(box[1, 0], box[2, 0], box[1, 1], box[2, 1])
    return vec


def box_to_center_points(box: np.ndarray) -> float:
    """Given the rectangular box. Will compute the center as the midpoint of a diagonal."""
    center_row = np.mean([box[0, 0], box[2, 0]])
    center_col = np.mean([box[0, 1], box[2, 1]])
    return center_row, center_col


def axis_from_mask(mask: np.ndarray) -> np.ndarray:
    """Given a folder path. Will import the mask and determine it's long axis."""
    # insert borders to the mask
    border = 10
    mask_mod = insert_borders(mask, border)
    # find contour
    mask_thresh_blur = ndimage.gaussian_filter(mask_mod, 1)
    cnts = measure.find_contours(mask_thresh_blur, 0.75)[0].astype(np.int32)
    # find minimum area bounding rectangle
    rect = cv2.minAreaRect(cnts)
    box = np.int0(cv2.boxPoints(rect))
    vec = box_to_unit_vec(box)
    center_row, center_col = box_to_center_points(box)
    return center_row, center_col, vec


# def rotate_results():
#     return

# def transform_coordinate_system():
# --> go to microns, and then also move origin etc. as needed
