import os
import tifffile as tiff
import numpy as np
from n2v.models import N2VConfig, N2V
from csbdeep.utils import plot_history
from scipy.ndimage import gaussian_filter
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import logging
from utils import set_logger, get_args


def setup_paths(main_path):
    paths = {
        "filaments_input": os.path.join(main_path, "z_projection", "filaments"),
        "puromycin_input": os.path.join(main_path, "z_projection", "puromycin"),
        "filaments_output": os.path.join(main_path, "n2v1", "filaments"),
        "puromycin_output": os.path.join(main_path, "n2v1", "puromycin")
    }
    return paths


def load_image(image_path):
    image = tiff.imread(image_path)
    return image[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions


def configure_model(image, output_folder, model_name="noise2void_model"):
    """Set up and initialize the N2V model configuration."""
    config = N2VConfig(image,
                       unet_kern_size=5,
                       train_steps_per_epoch=200,
                       train_epochs=50,
                       train_loss='mse',
                       batch_norm=True,
                       train_batch_size=16,
                       n2v_perc_pix=0.198,
                       n2v_patch_shape=(64, 64),
                       unet_n_first=16,
                       unet_residual=True,
                       n2v_manipulator='uniform_withCP',
                       n2v_neighborhood_radius=5,
                       single_net_per_channel=False,
                       )
    model = N2V(config, model_name, basedir=output_folder)
    model.prepare_for_training(metrics=())
    return model


def train_and_predict(model, image):
    """Train the model on the image and return the denoised prediction."""
    history = model.train(image, image)
    plot_training_history(history)

    # Predict the denoised image
    prediction = model.predict(image, axes='SYXC')
    return prediction[0, ..., 0]  # Return single channel


def plot_training_history(history):
    # Display the training history
    plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'])
    plt.title(f'Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()  # Optionally display the plot
    plt.close()


def filter_and_save_denoised_image(image, output_path):
    """Apply Gaussian filter and save the denoised image."""
    filtered_image = gaussian_filter(image, sigma=1.0)
    tiff.imwrite(output_path, filtered_image.astype(np.float32))


def process_folder(input_folder, output_folder):
    """Process all images in a given folder for denoising."""
    file_names = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    image_paths = [os.path.join(input_folder, f) for f in file_names]

    for image_name, image_path in zip(file_names, image_paths):
        image = load_image(image_path)
        model = configure_model(image, output_folder, model_name=f'noise2void_{image_name}_test')
        denoised_image = train_and_predict(model, image)
        output_path = os.path.join(output_folder, f"denoised_{image_name}")
        filter_and_save_denoised_image(denoised_image, output_path)
        tf.keras.backend.clear_session()
        gc.collect()


def main():

    set_logger()
    logging.info('logger set')

    arg_names = ['main_path']
    args = get_args(arg_names)
    main_path = args['main_path']

    paths = setup_paths(main_path)

    process_folder(paths["filaments_input"], paths["filaments_output"])
    process_folder(paths["puromycin_input"], paths["puromycin_output"])


if __name__ == "__main__":
    main()
