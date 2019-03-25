"""
Helper methods for converting pngs to gifs.
"""
import imageio
import glob


def convert_imgs_to_gif(experiment_dir, prefix):
    """
    Given an experiment_dir and prefix, converts all pngs matching that prefix
    to a gif and saves the gif to '<experiment_dir>/<prefix>.gif'.
    """
    images = []
    filenames = sorted(glob.glob("{}/{}*.png".format(experiment_dir, prefix)))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(
        "{}/{}.gif".format(experiment_dir, prefix),
        images,
        duration=1)
