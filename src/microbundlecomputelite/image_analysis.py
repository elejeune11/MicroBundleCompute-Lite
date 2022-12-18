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
from skimage.transform import rotate
from typing import List, Tuple, Union


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


def adjust_feature_param_dicts(feature_params: dict, img_uint8: np.ndarray, mask: np.ndarray, min_coverage: Union[float, int] = 100) -> dict:
    """Given feature parameters, an image, and a mask. Will automatically update the feature quality to ensure sufficient coverage.
    (min_coverage refers to the number of pixels that should be attributed to 1 tracking point)"""
    track_points_0 = mask_to_track_points(img_uint8, mask, feature_params)
    coverage = np.sum(mask) / track_points_0.shape[0]
    iter = 0
    qualityLevel = feature_params["qualityLevel"]
    while coverage > min_coverage and iter < 10:
        qualityLevel = qualityLevel * 10 ** (np.log10(0.1) / 10)  # this value raised to 10 is 0.1, so it will lower quality by an order of magnitude in 10 iterations
        feature_params["qualityLevel"] = qualityLevel
        track_points_0 = mask_to_track_points(img_uint8, mask, feature_params)
        coverage = np.sum(mask) / track_points_0.shape[0]
        iter += 1
    return feature_params


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
    feature_params = adjust_feature_param_dicts(feature_params, img_0, mask)
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


def save_tracking(*, folder_path: Path, tracker_row_all: List, tracker_col_all: List, info: np.ndarray = None, is_rotated: bool = False, rot_info: np.ndarray = None, is_translated: bool = False, fname: str = None) -> List:
    """Given tracking results. Will save as text files."""
    new_path = create_folder(folder_path, "results")
    num_beats = len(tracker_row_all)
    saved_paths = []
    for kk in range(0, num_beats):
        if fname is not None:
            file_path = new_path.joinpath(fname + "_beat%i_row.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all[kk])
            file_path = new_path.joinpath(fname + "_beat%i_col.txt" % (kk)).resolve()
            np.savetxt(str(file_path), tracker_col_all[kk])
            saved_paths.append(file_path)
        elif is_translated and is_rotated:
            file_path = new_path.joinpath("rotated_translated_beat%i_row.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all[kk])
            file_path = new_path.joinpath("rotated_translated_beat%i_col.txt" % (kk)).resolve()
            np.savetxt(str(file_path), tracker_col_all[kk])
            saved_paths.append(file_path)
        elif is_translated:
            file_path = new_path.joinpath("translated_beat%i_row.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all[kk])
            file_path = new_path.joinpath("translated_beat%i_col.txt" % (kk)).resolve()
            np.savetxt(str(file_path), tracker_col_all[kk])
            saved_paths.append(file_path)
        elif is_rotated:
            file_path = new_path.joinpath("rotated_beat%i_row.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all[kk])
            file_path = new_path.joinpath("rotated_beat%i_col.txt" % (kk)).resolve()
            np.savetxt(str(file_path), tracker_col_all[kk])
            saved_paths.append(file_path)
        else:
            file_path = new_path.joinpath("beat%i_row.txt" % (kk)).resolve()
            saved_paths.append(file_path)
            np.savetxt(str(file_path), tracker_row_all[kk])
            file_path = new_path.joinpath("beat%i_col.txt" % (kk)).resolve()
            np.savetxt(str(file_path), tracker_col_all[kk])
            saved_paths.append(file_path)
    if info is not None:
        file_path = new_path.joinpath("info.txt").resolve()
        np.savetxt(str(file_path), info)
    if rot_info is not None:
        file_path = new_path.joinpath("rot_info.txt").resolve()
        np.savetxt(str(file_path), rot_info)
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
    saved_paths = save_tracking(folder_path=folder_path, tracker_col_all=tracker_0_all, tracker_row_all=tracker_1_all, info=info)
    return saved_paths


def load_tracking_results(*, folder_path: Path, is_rotated: bool = False, is_translated: bool = False, fname: str = None) -> List:
    """Given the folder path. Will load tracking results. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("results").resolve()
    if res_folder_path.exists() is False:
        raise FileNotFoundError("tracking results are not present -- tracking must be run before visualization")
    rev_file_0 = res_folder_path.joinpath("rotated_beat0_col.txt").resolve()
    if is_rotated:
        if rev_file_0.is_file() is False:
            raise FileNotFoundError("rotated tracking results are not present -- rotated tracking must be run before rotated visualization")
    num_files = len(glob.glob(str(res_folder_path) + "/beat*.txt"))
    num_beats = int((num_files) / 2)
    tracker_row_all = []
    tracker_col_all = []
    for kk in range(0, num_beats):
        if fname is not None:
            tracker_row = np.loadtxt(str(res_folder_path) + "/" + fname + "_beat%i_row.txt" % (kk))
            tracker_col = np.loadtxt(str(res_folder_path) + "/" + fname + "_beat%i_col.txt" % (kk))
        elif is_rotated and is_translated:
            tracker_row = np.loadtxt(str(res_folder_path) + "/rotated_translated_beat%i_row.txt" % (kk))
            tracker_col = np.loadtxt(str(res_folder_path) + "/rotated_translated_beat%i_col.txt" % (kk))
        elif is_translated:
            tracker_row = np.loadtxt(str(res_folder_path) + "/translated_beat%i_row.txt" % (kk))
            tracker_col = np.loadtxt(str(res_folder_path) + "/translated_beat%i_col.txt" % (kk))
        elif is_rotated:
            tracker_row = np.loadtxt(str(res_folder_path) + "/rotated_beat%i_row.txt" % (kk))
            tracker_col = np.loadtxt(str(res_folder_path) + "/rotated_beat%i_col.txt" % (kk))
        else:
            tracker_row = np.loadtxt(str(res_folder_path) + "/beat%i_row.txt" % (kk))
            tracker_col = np.loadtxt(str(res_folder_path) + "/beat%i_col.txt" % (kk))
        tracker_row_all.append(tracker_row)
        tracker_col_all.append(tracker_col)
    info = np.loadtxt(str(res_folder_path) + "/info.txt")
    info_reshape = np.reshape(info, (-1, 3))
    if is_rotated:
        rot_info = np.loadtxt(str(res_folder_path) + "/rot_info.txt")
    else:
        rot_info = None
    return tracker_row_all, tracker_col_all, info_reshape, rot_info


def get_title_fname(kk: int, beat: int, is_rotated: bool = False, include_interp: bool = False) -> str:
    if is_rotated and include_interp:
        ti = "rotated frame %i, beat %i, with interpolation" % (kk, beat)
        fn = "rotated_%04d_disp_with_interp.png" % (kk)
        fn_gif = "rotated_abs_disp_with_interp.gif"
    elif is_rotated:
        ti = "rotated frame %i, beat %i" % (kk, beat)
        fn = "rotated_%04d_disp.png" % (kk)
        fn_gif = "rotated_abs_disp.gif"
    elif include_interp:
        ti = "frame %i, beat %i, with interpolation" % (kk, beat)
        fn = "%04d_disp_with_interp.png" % (kk)
        fn_gif = "abs_disp_with_interp.gif"
    else:
        ti = "frame %i, beat %i" % (kk, beat)
        fn = "%04d_disp.png" % (kk)
        fn_gif = "abs_disp.gif"
    return ti, fn, fn_gif


def create_pngs(
    folder_path: Path,
    tiff_list: List,
    tracker_row_all: List,
    tracker_col_all: List,
    info: np.ndarray,
    col_max: Union[float, int],
    col_map: object,
    *,
    is_rotated: bool = False,
    include_interp: bool = False,
    interp_tracker_row_all: List = None,
    interp_tracker_col_all: List = None
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
        if include_interp:
            interp_tracker_row = interp_tracker_row_all[beat]
            interp_tracker_col = interp_tracker_col_all[beat]
            _, interp_disp_all = compute_abs_position_timeseries(interp_tracker_row, interp_tracker_col)
        start_idx = int(info[beat, 1])
        end_idx = int(info[beat, 2])
        for kk in range(start_idx, end_idx):
            ti, fn, _ = get_title_fname(kk, beat, is_rotated, include_interp)
            plt.figure()
            plt.imshow(tiff_list[kk], cmap=plt.cm.gray)
            jj = kk - start_idx
            plt.scatter(tracker_col[:, jj], tracker_row[:, jj], c=disp_all[:, jj], s=10, cmap=col_map, vmin=0, vmax=col_max)
            if include_interp:
                plt.scatter(interp_tracker_col[:, jj], interp_tracker_row[:, jj], c=interp_disp_all[:, jj], s=7, cmap=col_map, vmin=0, vmax=col_max, linewidths=1, edgecolors=(0, 0, 0))
            plt.title(ti)
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().labelpad = 15
            cbar.set_label("absolute displacement (pixels)", rotation=270)
            plt.axis("off")
            path = pngs_folder_path.joinpath(fn).resolve()
            plt.savefig(str(path))
            plt.close()
            path_list.append(path)
    return path_list


def create_gif(folder_path: Path, png_path_list: List, is_rotated: bool = False, include_interp: bool = False) -> Path:
    """Given the pngs path list. Will create a gif."""
    img_list = []
    for pa in png_path_list:
        img_list.append(imageio.imread(pa))
    _, _, fn_gif = get_title_fname(0, 0, is_rotated, include_interp)
    gif_path = folder_path.joinpath("visualizations").resolve().joinpath(fn_gif).resolve()
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
    tracker_row_all, tracker_col_all, info, _ = load_tracking_results(folder_path=folder_path)
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


def interpolate_pos_from_tracking_arrays(
    tracker_row: np.ndarray,
    tracker_col: np.ndarray,
    row_col_sample: np.ndarray,
) -> np.ndarray:
    """Given tracking results for one beat and sample locations.
    Will return interpolated tracking results at the sample points."""
    num_frames = tracker_row.shape[1]
    num_sample_pts = row_col_sample.shape[0]
    row_sample = np.zeros((num_sample_pts, num_frames))
    col_sample = np.zeros((num_sample_pts, num_frames))
    row_sample[:, 0] = row_col_sample[:, 0]
    col_sample[:, 0] = row_col_sample[:, 1]
    row_col_pos = np.hstack((tracker_row[:, 0].reshape((-1, 1)), tracker_col[:, 0].reshape((-1, 1))))
    for kk in range(1, num_frames):
        row_col_vals = np.hstack((tracker_row[:, kk].reshape((-1, 1)) - tracker_row[:, 0].reshape((-1, 1)), tracker_col[:, kk].reshape((-1, 1)) - tracker_col[:, 0].reshape((-1, 1))))
        row_col_sample_vals = interpolate_points(row_col_pos, row_col_vals, row_col_sample)
        row_sample[:, kk] = row_col_sample_vals[:, 0] + row_col_sample[:, 0]
        col_sample[:, kk] = row_col_sample_vals[:, 1] + row_col_sample[:, 1]
    return row_sample, col_sample


def interpolate_pos_from_tracking_lists(
    tracker_row_all: List,
    tracker_col_all: List,
    row_col_sample: np.ndarray,
) -> List:
    """Given tracking results in a list and interpolation sample points. Will interpolate for all frames."""
    row_sample_list = []
    col_sample_list = []
    num_beats = len(tracker_row_all)
    for kk in range(0, num_beats):
        row_sample, col_sample = interpolate_pos_from_tracking_arrays(tracker_row_all[kk], tracker_col_all[kk], row_col_sample)
        row_sample_list.append(row_sample)
        col_sample_list.append(col_sample)
    return row_sample_list, col_sample_list


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


def rot_vec_to_rot_mat_and_angle(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    """Given a rotation vector. Will return a rotation matrix and rotation angle."""
    ang = np.arctan2(vec[0], vec[1])
    rot_mat = np.asarray([[np.cos(ang), -1.0 * np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return (rot_mat, ang)


def rot_image(
    img: np.ndarray,
    center_row: Union[float, int],
    center_col: Union[float, int],
    ang: float
) -> np.ndarray:
    """Given an image and rotation information. Will return rotated image."""
    new_img = rotate(img, ang / (np.pi) * 180, center=(center_col, center_row))
    return new_img


def rotate_points(
    row_pts: np.ndarray,
    col_pts: np.ndarray,
    rot_mat: np.ndarray,
    center_row: Union[float, int],
    center_col: Union[float, int]
) -> np.ndarray:
    """Given array vectors of points, rotaton matrix, and point to rotate about.
    Will perform rotation and return rotated points"""
    row_pts_centered = row_pts - center_row
    col_pts_centered = col_pts - center_col
    pts = np.hstack((row_pts_centered.reshape((-1, 1)), col_pts_centered.reshape((-1, 1)))).T
    pts_rotated = rot_mat @ pts
    new_row_pts = pts_rotated[0, :] + center_row
    new_col_pts = pts_rotated[1, :] + center_col
    return new_row_pts, new_col_pts


def rotate_points_array(
    row_pts_array: np.ndarray,
    col_pts_array: np.ndarray,
    rot_mat: np.ndarray,
    center_row: Union[float, int],
    center_col: Union[float, int]
) -> np.ndarray:
    """Given an array of row points and column points. Will rotate the whole array."""
    rot_row_pts_array = np.zeros(row_pts_array.shape)
    rot_col_pts_array = np.zeros(col_pts_array.shape)
    num_steps = row_pts_array.shape[1]
    for kk in range(0, num_steps):
        row_pts = row_pts_array[:, kk]
        col_pts = col_pts_array[:, kk]
        rot_row_pts, rot_col_pts = rotate_points(row_pts, col_pts, rot_mat, center_row, center_col)
        rot_row_pts_array[:, kk] = rot_row_pts
        rot_col_pts_array[:, kk] = rot_col_pts
    return rot_row_pts_array, rot_col_pts_array


def rotate_pts_all(
    row_pts_array_list: List,
    col_pts_array_list: List,
    rot_mat: np.ndarray,
    center_row: Union[float, int],
    center_col: Union[float, int]
) -> np.ndarray:
    """Given a list of row and column point arrays. Will rotate all of them."""
    rot_row_pts_array_list = []
    rot_col_pts_array_list = []
    num_arrays = len(row_pts_array_list)
    for kk in range(0, num_arrays):
        row_pts_array = row_pts_array_list[kk]
        col_pts_array = col_pts_array_list[kk]
        rot_row_pts_array, rot_col_pts_array = rotate_points_array(row_pts_array, col_pts_array, rot_mat, center_row, center_col)
        rot_row_pts_array_list.append(rot_row_pts_array)
        rot_col_pts_array_list.append(rot_col_pts_array)
    return rot_row_pts_array_list, rot_col_pts_array_list


def rotate_imgs_all(
    tiff_list: List,
    ang: float,
    center_row: Union[float, int],
    center_col: Union[float, int]
) -> np.ndarray:
    """Given tiff_list and rotation info. Will return all images rotated."""
    rot_tiff_list = []
    for kk in range(0, len(tiff_list)):
        rot_img = rot_image(tiff_list[kk], center_row, center_col, ang)
        rot_tiff_list.append(rot_img)
    return rot_tiff_list


def get_rotation_info(
    *,
    center_row_input: Union[float, int] = None,
    center_col_input: Union[float, int] = None,
    vec_input: np.ndarray = None,
    mask: np.ndarray = None
) -> Tuple[Union[float, int], Union[float, int], np.ndarray, Union[float, int]]:
    """Given either prescribed rotation or mask.
    Will compute rotation information (rotation matrix and angle).
    Prescribed rotation will override rotation computed by the mask."""
    if mask is not None:
        center_row, center_col, vec = axis_from_mask(mask)
    if center_row_input is not None:
        center_row = center_row_input
    if center_col_input is not None:
        center_col = center_col_input
    if vec_input is not None:
        vec = vec_input
    (rot_mat, ang) = rot_vec_to_rot_mat_and_angle(vec)
    return (center_row, center_col, rot_mat, ang, vec)


def run_rotation(
    folder_path: Path,
    input_mask: bool = True,
    *,
    center_row_input: Union[float, int] = None,
    center_col_input: Union[float, int] = None,
    vec_input: np.ndarray = None
) -> List:
    """Given rotation information. Will rotate the points according to the provided information."""
    if input_mask:
        mask_file_path = folder_path.joinpath("masks").resolve().joinpath("tissue_mask.txt").resolve()
        mask = read_txt_as_mask(mask_file_path)
        (center_row, center_col, rot_mat, _, vec) = get_rotation_info(center_row_input=center_row_input, center_col_input=center_col_input, vec_input=vec_input, mask=mask)
    else:
        (center_row, center_col, rot_mat, _, vec) = get_rotation_info(center_row_input=center_row_input, center_col_input=center_col_input, vec_input=vec_input)
    # read tracking results
    tracker_row_all, tracker_col_all, _, _ = load_tracking_results(folder_path=folder_path)
    # perform rotation
    rot_tracker_row_all, rot_tracker_col_all = rotate_pts_all(tracker_row_all, tracker_col_all, rot_mat, center_row, center_col)
    # save rotation info
    rot_info = np.asarray([[center_row, center_col], [vec[0], vec[1]]])
    # save rotation
    saved_paths = save_tracking(folder_path=folder_path, tracker_col_all=rot_tracker_col_all, tracker_row_all=rot_tracker_row_all, is_rotated=True, rot_info=rot_info)
    return saved_paths


def run_rotation_visualization(folder_path: Path, col_max: Union[int, float] = 10, col_map: object = plt.cm.viridis) -> List:
    """Given a folder path where rotated tracking has already been run. Will save visualizations."""
    # read image files
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = image_folder_to_path_list(movie_folder_path)
    tiff_list = read_all_tiff(name_list_path)
    # read rotated tracking results
    tracker_row_all, tracker_col_all, info, rot_info = load_tracking_results(folder_path=folder_path, is_rotated=True)
    # rotate tiffs
    center_row = rot_info[0, 0]
    center_col = rot_info[0, 1]
    vec = np.asarray([rot_info[1, 0], rot_info[1, 1]])
    (_, ang) = rot_vec_to_rot_mat_and_angle(vec)
    rot_tiff_list = rotate_imgs_all(tiff_list, ang, center_row, center_col)
    # create pngs
    png_path_list = create_pngs(folder_path, rot_tiff_list, tracker_row_all, tracker_col_all, info, col_max, col_map, is_rotated=True)
    # create gif
    gif_path = create_gif(folder_path, png_path_list, True)
    return png_path_list, gif_path


def scale_array_in_list(tracker_list: List, new_origin: Union[int, float], scale_mult: Union[int, float]) -> List:
    """Given a list of arrays of coordinates, new origin (in pixel coordinates), and new scale. Will subtract the origin and then multiply by the scale."""
    updated_tracker_list = []
    num_beats = len(tracker_list)
    for kk in range(0, num_beats):
        val_array = tracker_list[kk]
        new_val_array = (val_array - new_origin) * scale_mult
        updated_tracker_list.append(new_val_array)
    return updated_tracker_list


def run_scale_and_center_coordinates(
    folder_path: Path,
    pixel_origin_row: Union[int, float],
    pixel_origin_col: Union[int, float],
    microns_per_pixel_row: Union[int, float],
    microns_per_pixel_col: Union[int, float],
    use_rotated: bool = False,
    fname: str = None,
    new_fname: str = None
) -> List:
    """Given information to transform the coordinate system (translation only). """
    tracker_row_all, tracker_col_all, _, _ = load_tracking_results(folder_path=folder_path, is_rotated=use_rotated, fname=fname)
    updated_tracker_row_all = scale_array_in_list(tracker_row_all, pixel_origin_row, microns_per_pixel_row)
    updated_tracker_col_all = scale_array_in_list(tracker_col_all, pixel_origin_col, microns_per_pixel_col)
    saved_paths = save_tracking(folder_path=folder_path, tracker_col_all=updated_tracker_col_all, tracker_row_all=updated_tracker_row_all, is_translated=True, is_rotated=use_rotated, fname=new_fname)
    return saved_paths


def run_interpolate(
    folder_path: Path,
    row_col_sample: np.ndarray,
    interpolation_fname: str = "interpolation",
    is_rotated: bool = False,
    is_translated: bool = False
) -> List:
    """Given a folder path, information, and sample points. Will compute and save interpolation at the sample points."""
    # load tracking results
    tracker_row_all, tracker_col_all, _, _ = load_tracking_results(folder_path=folder_path, is_rotated=is_rotated, is_translated=is_translated)
    # perform interpolation of tracking results
    row_sample_list, col_sample_list = interpolate_pos_from_tracking_lists(tracker_row_all, tracker_col_all, row_col_sample)
    # save interpolated results
    saved_paths = save_tracking(folder_path=folder_path, tracker_col_all=col_sample_list, tracker_row_all=row_sample_list, fname=interpolation_fname)
    return saved_paths


def visualize_interpolate(
    folder_path: Path,
    *,
    is_rotated: bool = False,
    is_translated: bool = False,
    interpolation_fname: str = "interpolation",
    col_max: Union[int, float] = 10,
    col_map: object = plt.cm.viridis
) -> List:
    """Given folder path and plotting information. Will run and save visualization."""
    # read image files and tracking results
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = image_folder_to_path_list(movie_folder_path)
    tiff_list = read_all_tiff(name_list_path)
    tracker_row_all, tracker_col_all, info, rot_info = load_tracking_results(folder_path=folder_path, is_rotated=is_rotated, is_translated=is_translated)
    if is_rotated:
        center_row = rot_info[0, 0]
        center_col = rot_info[0, 1]
        vec = np.asarray([rot_info[1, 0], rot_info[1, 1]])
        (_, ang) = rot_vec_to_rot_mat_and_angle(vec)
        tiff_list = rotate_imgs_all(tiff_list, ang, center_row, center_col)
    # read interpolated results
    interp_tracker_row_all, interp_tracker_col_all, _, _ = load_tracking_results(folder_path=folder_path, is_rotated=is_rotated, is_translated=is_translated, fname=interpolation_fname)
    # create pngs
    png_path_list = create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, col_max, col_map, is_rotated=is_rotated, include_interp=True, interp_tracker_row_all=interp_tracker_row_all, interp_tracker_col_all=interp_tracker_col_all)
    # create gif
    gif_path = create_gif(folder_path, png_path_list, is_rotated, True)
    return png_path_list, gif_path
