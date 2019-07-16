"""Functions for file handling."""

import plyfile
import pandas as pd
import vaex as vx
import logging

logger = logging.getLogger('ectopylasm.io')
logger.setLevel(logging.DEBUG)


def load_plyfile(filename):
    """Load a PLY file."""
    plydata = plyfile.PlyData.read(filename)
    return plydata


def vertex_dict_from_plyfile(filename):
    """Load vertices from plyfile and return as dict."""
    plydata = load_plyfile(filename)
    xyz = dict(x=plydata['vertex']['x'], y=plydata['vertex']['y'], z=plydata['vertex']['z'])
    return xyz


def pandas_vertices_from_plyfile(filename):
    """Load vertices from plyfile and return as pandas DataFrame."""
    xyz = vertex_dict_from_plyfile(filename)
    return pd.DataFrame(xyz)


def vaex_vertices_from_plyfile(filename):
    """Load vertices from plyfile and return as vaex DataFrame."""
    xyz = vertex_dict_from_plyfile(filename)
    return vx.from_dict(xyz)
