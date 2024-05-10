import logging
import os
from pathlib import Path

import pandas as pd
import pytest
from plots import Plots


@pytest.fixture
def plot_class():
    yield Plots('https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json', 'plots')


def test_init_types(plot_class):
    assert isinstance(plot_class._Plots__df, pd.DataFrame)
    assert isinstance(plot_class._Plots__paths, dict)
    assert isinstance(plot_class._Plots__folder_to_save_pictures, str)
    assert isinstance(plot_class._Plots__path_folder_to_save_pictures, Path)


def test_get_paths_type(plot_class):
    paths = plot_class.get_paths()
    assert isinstance(paths, dict)


def test_get_paths(plot_class):
    paths = plot_class.get_paths()
    assert paths == plot_class._Plots__paths


def test_confusion_matrix_save_in_right_place(plot_class, mocker):
    file_path = plot_class.get_paths()['confusion_matrix']

    mocker.patch('matplotlib.pyplot.show', return_value=None)

    if os.path.exists(file_path):
        os.remove(file_path)

    assert os.path.exists(file_path) is False

    plot_class.draw_conf_mat_plot(True)

    assert os.path.isfile(file_path)


def test_amount_rooms_save_in_right_place(plot_class, mocker):
    file_path = plot_class.get_paths()['amount_rooms']

    mocker.patch('matplotlib.pyplot.show', return_value=None)

    if os.path.exists(file_path):
        os.remove(file_path)

    assert os.path.exists(file_path) is False

    plot_class.draw_amount_room_plot(True)

    assert os.path.isfile(file_path)


os.system('pytest test.py')


