import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from torchvision.utils import make_grid

from .utils import scale2range


with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "oscar_cmap.pkl"), "rb") as f:  
    OSCAR_CMAP = LinearSegmentedColormap.from_list("oscar_cmap", pickle.load(f))


def visualise_samples(x, scale=False):
    while len(x.shape) < 4:
        x = x.unsqueeze(0)
    if scale:
        x = scale2range(x, [0, 1])
    fig = plt.figure()
    grid = make_grid(x).permute(1, 2, 0).cpu()
    if x.shape[1] == 1:
        grid = grid[:, :, 0]
    plt.imshow(grid, cmap=OSCAR_CMAP)
    return fig
