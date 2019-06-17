#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the io module.
"""
# import pytest

from ectopylasm import io


def test_load_plyfile():
    plydata = io.load_plyfile('data/cube.ply')

    assert plydata['x'] == [-1, 1, 1, -1, -1, 1, 1, -1]
    assert plydata['y'] == [-1, -1, 1, 1, -1, -1, 1, 1]
    assert plydata['z'] == [-1, -1, -1, -1, 1, 1, 1, 1]
