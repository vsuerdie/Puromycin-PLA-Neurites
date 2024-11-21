import numpy as np
import cv2
import tifffile as tiff
import os
from glob import glob
import logging
from utils import set_logger, get_args


def load_mask_images(directory_path):
    tif_files_paths = glob(os.path.join(directory_path, '*.tif'))
    masks = [tiff.imread(p) for p in tif_files_paths]
    images_names = [os.path.splitext(os.path.basename(p))[0] for p in tif_files_paths]

    return masks, images_names


def get_closed_holes_masks(masks, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_masks = [cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) for mask in masks]
    return closed_masks


def get_dilated_masks(masks, kernel_size):
    circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_masks = [cv2.dilate(mask, circle_kernel, iterations=1) for mask in masks]
    return dilated_masks


def filter_out_small_filaments(masks, min_size):
    filtered_masks = []
    for mask in masks:
        binary_mask = (mask == 2).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(binary_mask)
        filtered_mask = np.zeros_like(mask)

        for label in range(1, num_labels):
            if np.sum(labels_im == label) >= min_size:
                filtered_mask[labels_im == label] = 2

        filtered_masks.append(filtered_mask)

    return filtered_masks


def get_filaments_minus_dapi(filaments_masks, dapi_masks):
    filaments_minus_dapi_masks = [np.where(dapi_mask == 2, 1, filaments_mask) for filaments_mask, dapi_mask in zip(filaments_masks, dapi_masks)]
    return filaments_minus_dapi_masks


def save_final_masks(final_masks, filaments_names, output_path):
    os.makedirs(output_path, exist_ok=True)
    if len(final_masks) != len(filaments_names):
        raise ValueError("The number of masks and names must be the same")

    for mask, name in zip(final_masks, filaments_names):
        tiff.imwrite(os.path.join(output_path, name.replace('denoised', 'mask')), mask)


def main():

    set_logger()
    logging.info('logger set')

    arg_names = ['input_dir_dapi_masks',
                 'input_dir_filaments_masks',
                 'output_path',
                 'closed_masks_kernel_size',
                 'dapi_dilation_kernel_size',
                 'min_filaments_size',
                 'filaments_dilation_kernel_size']

    args = get_args(arg_names)
    input_dir_dapi_masks = args['input_dir_dapi_masks']
    input_dir_filaments_masks = args['input_dir_filaments_masks']
    output_path = args['output_paths']
    closed_masks_kernel_size = int(args['closed_masks_kernel_size'])
    dapi_dilation_kernel_size = int(args['dapi_dilation_kernel_size'])
    min_filaments_size = int(args['min_filaments_size'])
    filaments_dilation_kernel_size = int(args['min_filaments_kernel_size'])

    dapi_masks, dapi_names = load_mask_images(input_dir_dapi_masks)
    filaments_masks, filaments_names = load_mask_images(input_dir_filaments_masks)
    dapi_masks_closed = get_closed_holes_masks(dapi_masks, closed_masks_kernel_size)
    filaments_masks_closed = get_closed_holes_masks(filaments_masks, closed_masks_kernel_size)
    dapi_masks_dilated = get_dilated_masks(dapi_masks_closed, kernel_size=dapi_dilation_kernel_size)
    filaments_masks_filtered = filter_out_small_filaments(filaments_masks_closed, min_size=min_filaments_size)
    filaments_masks_dilated = get_dilated_masks(filaments_masks_filtered, kernel_size=filaments_dilation_kernel_size)
    filaments_masks_closed_2nd = get_closed_holes_masks(filaments_masks_dilated, closed_masks_kernel_size)
    filaments_minus_dapi_masks = get_filaments_minus_dapi(filaments_masks_closed_2nd, dapi_masks_dilated)
    save_final_masks(filaments_minus_dapi_masks, filaments_names, output_path)

    if __name__ == "__main__":
        main()
