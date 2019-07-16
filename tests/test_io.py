#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the io module."""
import os
import numpy as np
from ectopylasm import io


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_load_plyfile():
    """Test io.load_plyfile."""
    plydata = io.load_plyfile(TEST_DIR + '/data/cube.ply')

    assert np.all(plydata['vertex']['x'] == [-1, 1, 1, -1, -1, 1, 1, -1])
    assert np.all(plydata['vertex']['y'] == [-1, -1, 1, 1, -1, -1, 1, 1])
    assert np.all(plydata['vertex']['z'] == [-1, -1, -1, -1, 1, 1, 1, 1])


def test_vertex_dict_from_plyfile():
    """Test io.vertex_dict_from_plyfile."""
    vertices = io.vertex_dict_from_plyfile(TEST_DIR + '/data/cube.ply')

    assert np.all(vertices['x'] == [-1, 1, 1, -1, -1, 1, 1, -1])
    assert np.all(vertices['y'] == [-1, -1, 1, 1, -1, -1, 1, 1])
    assert np.all(vertices['z'] == [-1, -1, -1, -1, 1, 1, 1, 1])


def test_pandas_vertices_from_plyfile():
    """Test io.pandas_vertices_from_plyfile."""
    vertices = io.pandas_vertices_from_plyfile(TEST_DIR + '/data/cube.ply')

    assert np.all(vertices['x'] == [-1, 1, 1, -1, -1, 1, 1, -1])
    assert np.all(vertices['y'] == [-1, -1, 1, 1, -1, -1, 1, 1])
    assert np.all(vertices['z'] == [-1, -1, -1, -1, 1, 1, 1, 1])


def test_vaex_vertices_from_plyfile():
    """Test io.vaex_vertices_from_plyfile."""
    vertices = io.vaex_vertices_from_plyfile(TEST_DIR + '/data/cube.ply')

    assert np.all(vertices['x'] == [-1, 1, 1, -1, -1, 1, 1, -1])
    assert np.all(vertices['y'] == [-1, -1, 1, 1, -1, -1, 1, 1])
    assert np.all(vertices['z'] == [-1, -1, -1, -1, 1, 1, 1, 1])
