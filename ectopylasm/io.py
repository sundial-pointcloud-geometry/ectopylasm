import plyfile
import pandas as pd
import vaex as vx
import logging

logger = logging.getLogger('ectopylasm.io')
logger.setLevel(logging.DEBUG)


def load_plyfile(filename):
    plydata = plyfile.PlyData.read(filename)
    return plydata


def vertex_dict_from_plyfile(filename):
    plydata = load_plyfile(filename)
    xyz = dict(x=plydata['vertex']['x'], y=plydata['vertex']['y'], z=plydata['vertex']['z'])
    return xyz


def pandas_vertices_from_plyfile(filename):
    xyz = vertex_dict_from_plyfile(filename)
    return pd.DataFrame(xyz)


def vaex_vertices_from_plyfile(filename):
    xyz = vertex_dict_from_plyfile(filename)
    return vx.from_dict(xyz)
