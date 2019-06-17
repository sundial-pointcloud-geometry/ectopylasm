import numpy as np
import logging

import ipyvolume as ipv
import pptk


logger = logging.getLogger('ectopylasm.visualize')
logger.setLevel(logging.INFO)


def random_sample(xyz, total, sample_frac):
    sample = np.random.choice(total, int(sample_frac * total), replace=False)
    logger.debug("sample size:", int(sample_frac * total), "out of total", total)
    return dict(x=xyz['x'][sample], y=xyz['y'][sample], z=xyz['z'][sample])


def ipv_plot_plydata(plydata, sample_frac=1, shape='circle2d', **kwargs):
    if sample_frac < 1:
        xyz = random_sample(plydata['vertex'], plydata['vertex'].count, sample_frac)
    else:
        xyz = dict(x=plydata['vertex']['x'], y=plydata['vertex']['y'], z=plydata['vertex']['z'])
    ipv.scatter(**xyz, shape=shape, **kwargs)
    ipv.show()


def pptk_plot_plydata(plydata, **kwargs):
    pptk.viewer(np.array([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T)


def ipv_plot_df(df, sample_frac=1, shape='circle2d', **kwargs):
    if sample_frac < 1:
        xyz = random_sample(df, len(df), sample_frac)
    else:
        xyz = dict(x=df['x'].values, y=df['y'].values, z=df['z'].values)
    ipv.scatter(**xyz, shape=shape, **kwargs)
    ipv.show()


def pptk_plot_df(df, **kwargs):
    pptk.viewer(np.array([df['x'], df['y'], df['z']]).T)
