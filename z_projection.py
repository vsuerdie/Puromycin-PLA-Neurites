import numpy as np
from readlif.reader import LifFile
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import logging
from utils import set_logger, get_args


def lif_to_numpy_list(lif_file_path):
    lif = LifFile(lif_file_path)
    image_list = []
    name_list = []

    for img in lif.get_iter_image():
        name_list.append(img.name)
        logging.info(f"Image name {img.name} appended")
        im_as_z_list_xyc = [np.dstack([np.array(img.get_frame(z, 0, c))
                                       for c in range(img.channels)]) for z in range(img.dims.z)]
        im_zxyc = np.stack(im_as_z_list_xyc, axis=0)
        image_list.append(im_zxyc)
        logging.info(f"Image {img.name} appended")

    logging.info('Images from lif returned as list with axis order zxyc')

    return image_list, name_list


def get_z_projection_all_channels(image_list):

    dapi_z_projection = []
    filaments_z_projection = []
    puromycin_z_projection = []

    for img in image_list:
        if img.shape[3] != 3:
            dapi_z_projection.append(np.max(img[:, :, :, 0], axis=0))
            filaments_z_projection.append(np.max(img[:, :, :, 1], axis=0))
            puromycin_z_projection.append(np.max(img[:, :, :, 2], axis=0))
        else:
            logging.warning("Image has less than 3 channels and will be skipped.")

    return dapi_z_projection, filaments_z_projection, puromycin_z_projection


def plot_all_images(projections_l_of_l, name_list, channels):

    for idx, name in enumerate(name_list):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i in range(len(axes)):
            axes[i].imshow(projections_l_of_l[i][idx], cmap='gray')
            axes[i].set_title(f'{channels[i]} - {name_list[idx]}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


def create_directories(main_path, channels):
    analysis_path = os.path.join(main_path, 'image_analysis', 'z_projection')
    paths = {c: os.path.join(analysis_path, c) for c in channels}
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def save_images(main_path, projections_l_of_l, name_list, channels):
    paths = create_directories(main_path, channels)

    for idx, name in enumerate(name_list):
        for projection, label in zip(projections_l_of_l, channels):
            if idx < len(projection):
                file_path = os.path.join(paths[label], f"{name}_{label}.tif")
                tiff.imwrite(file_path, projection[idx])


def main():

    set_logger()
    logging.info('logger set')

    arg_names = ['main_path', 'lif_file_path']
    args = get_args(arg_names)
    main_path = args['main_path']
    lif_file_path = args['lif_file_path']

    channels = ['dapi', 'filaments', 'puromycin']

    image_list, name_list = lif_to_numpy_list(lif_file_path)
    dapi_z_projection, filaments_z_projection, puromycin_z_projection = get_z_projection_all_channels(image_list)
    # plot_all_images([dapi_z_projection, filaments_z_projection, puromycin_z_projection], name_list, channels)
    save_images(main_path,
                [dapi_z_projection, filaments_z_projection, puromycin_z_projection],
                name_list,
                channels)


if __name__ == "__main__":
    main()
