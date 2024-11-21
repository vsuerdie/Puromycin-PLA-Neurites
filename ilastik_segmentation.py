import os
from glob import glob
import subprocess
import logging
from utils import set_logger, get_args

def run_ilastik_for_channel(ilastik_path, project_path, raw_data_path, output_path):
    raw_data_files = glob(os.path.join(raw_data_path, '*.tif'))

    if not raw_data_files:
        logging.warning(f"No .tif files found in {raw_data_path}.")
        return

    for input_file in raw_data_files:
        output_file = os.path.join(output_path, f"{os.path.basename(input_file)}.tif")
        cmd = (
            f'"{ilastik_path}" --headless --project="{project_path}" '
            f'--raw_data="{input_file}" --output_format=tif '
            f'--output_filename_format="{output_file}" '
            f'--export_source="Simple Segmentation"'
        )

        logging.info(f"Running command for {input_file}")
        result = subprocess.run(cmd, shell=True, capture_output=True)

        if result.returncode == 0:
            logging.info(f"Successfully processed {input_file}")
        else:
            logging.error(f"Error processing {input_file}: {result.stderr.decode('utf-8', errors='replace')}")


def main():

    set_logger()
    logging.info('logger set')
    arg_names = ['main_path']
    args = get_args(arg_names)
    main_path = args['main_path']

    base_path = args.base_path
    ilastik_path = args.ilastik_path
    # Configuration for each channel
    channels = {
        "dapi": {
            "project_path": os.path.join(main_path, "ilastik_segmentation", "projects", "dapi.ilp"),
            "raw_data_path": os.path.join(main_path, "z_projection", "dapi"),
            "output_path": os.path.join(main_path, "ilastik_segmentation", "results", "results_dapi")
        },
        "filaments": {
            "project_path": os.path.join(main_path, "ilastik_segmentation", "projects", "filaments.ilp"),
            "raw_data_path": os.path.join(main_path, "n2v", "filaments"),
            "output_path": os.path.join(main_path, "ilastik_segmentation", "results", "results_filaments")
        }
    }

    for channel, paths in channels.items():
        logging.info(f"Processing channel: {channel}")
        run_ilastik_for_channel(
            ilastik_path,
            paths["project_path"],
            paths["raw_data_path"],
            paths["output_path"]
        )

if __name__ == "__main__":
    main()
