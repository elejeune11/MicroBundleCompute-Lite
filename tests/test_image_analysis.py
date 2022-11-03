import matplotlib.pyplot as plt
from microbundlecomputelite import image_analysis as ia
import numpy as np
import os
from pathlib import Path
import pytest
from skimage import io


def test_hello_world():
    res = ia.hello_microbundle_compute()
    assert res == "Hello World!"


def files_path():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    return data_path


def example_path(example_name):
    data_path = files_path()
    ex_path = data_path.joinpath(example_name).resolve()
    return ex_path


def movie_path(example_name):
    ex_path = example_path(example_name)
    mov_path = ex_path.joinpath("movie").resolve()
    return mov_path


def tissue_mask_path(example_name):
    ex_path = example_path(example_name)
    mask_path = ex_path.joinpath("masks").resolve()
    t_m_path = mask_path.joinpath("tissue_mask.txt").resolve()
    return t_m_path


def test_read_tiff():
    mov_path = movie_path("real_example_super_short")
    img_path = mov_path.joinpath("ex1_0000.TIF").resolve()
    known = io.imread(img_path)
    found = ia.read_tiff(img_path)
    assert np.allclose(known, found)


def test_create_folder_guaranteed_conditions():
    folder_path = example_path("real_example_super_short")
    new_folder_name = "test_create_folder_%i" % (np.random.random() * 1000000)
    new_folder = ia.create_folder(folder_path, new_folder_name)
    assert new_folder.is_dir()
    new_folder = ia.create_folder(folder_path, new_folder_name)
    assert new_folder.is_dir()


def test_image_folder_to_path_list():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    assert len(name_list_path) == 5
    for kk in range(0, 5):
        assert os.path.isfile(name_list_path[kk])


def test_read_all_tiff():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    assert len(tiff_list) == 5
    assert tiff_list[0].shape == (512, 512)


def test_uint16_to_uint8():
    array_8 = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
    array_8[0, 0] = 0
    array_8[1, 0] = 255
    array_16 = array_8.astype(np.uint16) * 100
    found = ia.uint16_to_uint8(array_16)
    assert np.allclose(array_8, found)


def test_uint16_to_uint8_all():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    uint8_list = ia.uint16_to_uint8_all(tiff_list)
    for img in uint8_list:
        assert img.dtype is np.dtype('uint8')


def test_bool_to_uint8():
    arr_bool = np.random.random((10, 10)) > 0.5
    arr_uint8 = ia.bool_to_uint8(arr_bool)
    assert np.max(arr_uint8) == 1
    assert np.min(arr_uint8) == 0
    assert arr_uint8.dtype == np.dtype("uint8")


def test_read_txt_as_mask():
    file_path = tissue_mask_path("real_example_super_short")
    arr = ia.read_txt_as_mask(file_path)
    assert arr.dtype is np.dtype("uint8")


def test_get_tracking_param_dicts():
    feature_params, lk_params = ia.get_tracking_param_dicts()
    assert feature_params["maxCorners"] == 10000
    assert feature_params["qualityLevel"] == 0.1  # 0.005
    assert feature_params["minDistance"] == 5
    assert feature_params["blockSize"] == 5
    assert lk_params["winSize"][0] == 10
    assert lk_params["winSize"][1] == 10
    assert lk_params["maxLevel"] == 10
    assert lk_params["criteria"][1] == 10
    assert lk_params["criteria"][2] == 0.03


def test_mask_to_track_points():
    mov_path = movie_path("real_example_super_short")
    img_path = mov_path.joinpath("ex1_0000.TIF").resolve()
    img = ia.read_tiff(img_path)
    img_uint8 = ia.uint16_to_uint8(img)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params, _ = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8, mask, feature_params)
    assert track_points_0.shape[1] == 1
    assert track_points_0.shape[2] == 2


def test_mask_to_track_points_synthetic():
    img = np.zeros((100, 100))
    for kk in range(1, 10):
        img[int(kk * 10), int(kk * 5)] = 1
    img_uint8 = ia.uint16_to_uint8(img)
    mask = np.ones(img_uint8.shape)
    feature_params, _ = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8, mask, feature_params)
    tp_0_ix_col = np.sort(track_points_0[:, 0, 0])  # note col
    tp_0_ix_row = np.sort(track_points_0[:, 0, 1])  # note row
    for kk in range(1, 10):
        tp_0 = tp_0_ix_row[kk - 1]
        tp_1 = tp_0_ix_col[kk - 1]
        val0 = int(kk * 10)
        val1 = int(kk * 5)
        assert np.isclose(tp_0, val0, atol=1)
        assert np.isclose(tp_1, val1, atol=1)


def test_track_one_step():
    mov_path = movie_path("real_example_super_short")
    img_path = mov_path.joinpath("ex1_0000.TIF").resolve()
    img = ia.read_tiff(img_path)
    img_uint8_0 = ia.uint16_to_uint8(img)
    img_path = mov_path.joinpath("ex1_0001.TIF").resolve()
    img = ia.read_tiff(img_path)
    img_uint8_1 = ia.uint16_to_uint8(img)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    feature_params, lk_params = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8_0, mask, feature_params)
    track_points_1 = ia.track_one_step(img_uint8_0, img_uint8_1, track_points_0, lk_params)
    assert track_points_1.shape[1] == 1
    assert track_points_1.shape[2] == 2
    assert track_points_1.shape[0] == track_points_0.shape[0]
    compare = np.abs(track_points_1 - track_points_0)
    assert np.max(compare[:, 0, 0]) < lk_params["winSize"][0]
    assert np.max(compare[:, 0, 1]) < lk_params["winSize"][1]


def test_track_one_step_synthetic():
    img_0 = np.zeros((100, 100))
    for kk in range(1, 10):
        img_0[int(kk * 10), int(kk * 5)] = 1
    img_1 = np.zeros((100, 100))
    for kk in range(1, 10):
        img_1[int(kk * 10 + 1), int(kk * 5 + 1)] = 1
    img_uint8_0 = ia.uint16_to_uint8(img_0)
    img_uint8_1 = ia.uint16_to_uint8(img_1)
    mask = np.ones(img_uint8_0.shape)
    feature_params, lk_params = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8_0, mask, feature_params)
    track_points_1 = ia.track_one_step(img_uint8_0, img_uint8_1, track_points_0, lk_params)
    compare = np.abs(track_points_1 - track_points_0)
    assert np.max(compare) < np.sqrt(2)


def test_track_all_steps():
    folder_path = movie_path("real_example_super_short")
    name_list_path = ia.image_folder_to_path_list(folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    img_list_uint8 = ia.uint16_to_uint8_all(tiff_list)
    file_path = tissue_mask_path("real_example_super_short")
    mask = ia.read_txt_as_mask(file_path)
    tracker_0, tracker_1 = ia.track_all_steps(img_list_uint8, mask)
    diff_0 = np.abs(tracker_0[:, 0] - tracker_0[:, -1])
    diff_1 = np.abs(tracker_1[:, 0] - tracker_1[:, -1])
    _, lk_params = ia.get_tracking_param_dicts()
    assert np.max(diff_0) < lk_params["winSize"][0]
    assert np.max(diff_1) < lk_params["winSize"][1]


def test_compute_abs_position_timeseries():
    num_pts = 3
    num_frames = 100
    tracker_0 = 100 * np.ones((num_pts, num_frames)) + np.random.random((num_pts, num_frames))
    tracker_1 = 50 * np.ones((num_pts, num_frames)) + np.random.random((num_pts, num_frames))
    disp_abs_mean, disp_abs_all = ia.compute_abs_position_timeseries(tracker_0, tracker_1)
    assert disp_abs_mean.shape[0] == num_frames
    assert np.max(disp_abs_mean) < np.sqrt(2.0)
    assert disp_abs_all.shape[1] == num_frames


def test_get_time_segment_param_dicts():
    time_seg_params = ia.get_time_segment_param_dicts()
    assert time_seg_params["peakDist"] == 20


def test_compute_valleys():
    x = np.linspace(0, 500 * np.pi * 2.0, 500)
    timeseries = np.sin(x / (np.pi * 2.0) / 20 - np.pi / 2.0)
    info = ia.compute_valleys(timeseries)
    assert info.shape[0] == 2
    assert np.isclose(timeseries[info[0, 1]], -1, atol=.01)
    assert np.isclose(timeseries[info[0, 2]], -1, atol=.01)
    assert np.isclose(timeseries[info[1, 1]], -1, atol=.01)
    assert np.isclose(timeseries[info[1, 2]], -1, atol=.01)
    li = 10 * [-0.99] + list(timeseries) + 10 * [-0.99]
    timeseries = np.asarray(li)
    info = ia.compute_valleys(timeseries)
    assert np.isclose(timeseries[info[0, 1]], -1, atol=.01)
    assert np.isclose(timeseries[info[0, 2]], -1, atol=.01)
    assert np.isclose(timeseries[info[1, 1]], -1, atol=.01)
    assert np.isclose(timeseries[info[1, 2]], -1, atol=.01)


def test_split_tracking():
    tracker_0 = np.zeros((10, 100))
    tracker_1 = np.ones((10, 100))
    info = [[0, 10, 30], [1, 30, 35], [2, 35, 85]]
    info = np.asarray(info)
    tracker_0_all, tracker_1_all = ia.split_tracking(tracker_0, tracker_1, info)
    assert len(tracker_0_all) == 3
    assert len(tracker_1_all) == 3
    for kk in range(0, 3):
        assert tracker_0_all[kk].shape[0] == 10
        assert tracker_0_all[kk].shape[1] == info[kk, 2] - info[kk, 1]
        assert tracker_1_all[kk].shape[0] == 10
        assert tracker_1_all[kk].shape[1] == info[kk, 2] - info[kk, 1]


def test_save_tracking():
    tracker_0 = np.zeros((10, 100))
    tracker_1 = np.ones((10, 100))
    info = [[0, 10, 30], [1, 30, 35], [2, 35, 85]]
    info = np.asarray(info)
    tracker_0_all, tracker_1_all = ia.split_tracking(tracker_0, tracker_1, info)
    folder_path = example_path("real_example_super_short")
    saved_paths = ia.save_tracking(folder_path, tracker_0_all, tracker_1_all, info)
    for pa in saved_paths:
        assert pa.is_file()
    assert len(saved_paths) == info.shape[0] * 2 + 1


def test_run_tracking():
    folder_path = example_path("real_example_short")
    saved_paths = ia.run_tracking(folder_path)
    assert len(saved_paths) == 3
    for pa in saved_paths:
        assert pa.is_file()


def test_load_tracking_results():
    folder_path = example_path("real_example_short")
    _ = ia.run_tracking(folder_path)
    tracker_row_all, tracker_col_all, info = ia.load_tracking_results(folder_path)
    assert len(tracker_row_all) == 1
    assert len(tracker_row_all) == 1
    assert len(tracker_col_all) == 1
    assert len(tracker_col_all) == 1
    assert info.shape[1] == 3
    folder_path = example_path("fake_example_short")
    with pytest.raises(FileNotFoundError) as error:
        ia.load_tracking_results(folder_path)
    assert error.typename == "FileNotFoundError"


def test_create_pngs_gif_mp4():
    folder_path = example_path("real_example_short")
    _ = ia.run_tracking(folder_path)
    tracker_row_all, tracker_col_all, info = ia.load_tracking_results(folder_path)
    mov_path = movie_path("real_example_short")
    name_list_path = ia.image_folder_to_path_list(mov_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    col_max = 3
    col_map = plt.cm.viridis
    path_list = ia.create_pngs(folder_path, tiff_list, tracker_row_all, tracker_col_all, info, col_max, col_map)
    for pa in path_list:
        assert pa.is_file()
    gif_path = ia.create_gif(folder_path, path_list)
    assert gif_path.is_file()
    mp4_path = ia.create_mp4(folder_path, gif_path)
    assert mp4_path.is_file()


def test_run_visualization():
    folder_path = example_path("real_example_short")
    _ = ia.run_tracking(folder_path)
    col_max = 3
    col_map = plt.cm.viridis
    png_path_list, gif_path, mp4_path = ia.run_visualization(folder_path, col_max, col_map)
    for pa in png_path_list:
        assert pa.is_file()
    assert gif_path.is_file()
    assert mp4_path.is_file()
