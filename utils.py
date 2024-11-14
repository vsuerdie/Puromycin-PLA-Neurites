import logging
import argparse

def get_args(arg_list):
    parser = argparse.ArgumentParser()
    for arg in arg_list:
        parser.add_argument(f'--{arg}', required=True)
    return vars(parser.parse_args())



# def get_user_args():
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--main-path', type=str, default=None, help='Path to the folder in which is the data')
#     parser.add_argument('--lif-file-path', type=str, default=None, help='Path to the LIF file')
#     parser.add_argument('--input-dir-n2v-filaments', type=str, default=None, help='Input directory for z-projected filament images for n2v')
#     parser.add_argument('--input-dir-n2v-puromycin', type=str, default=None,help='Input directory for z-projected puromycin images for n2v')
#     parser.add_argument('--input-dir-dapi-masks', type=str, default=None, help='Input directory for DAPI masks')
#     parser.add_argument('--input-dir-filaments-masks', type=str, default=None, help='Input directory for filaments masks')
#     parser.add_argument('--input-dir-masks', type=str, default=None, help='Input directory for masks folder')
#     parser.add_argument('--csv-file', type=str, default=None, help="Path to the RS-FISH results csv file")
#     parser.add_argument('--output-path', type=str, default=None, help='Output path for the masks')
#     parser.add_argument('--path-macro', type=str, default=None, help='Path to the ImageJ macro')
#     parser.add_argument('--path-imagej', type=str, default=None, help='Path to the ImageJ executable')
#     parser.add_argument('--path-output-log', type=str, default=None, help='Path to the output log file')
#     parser.add_argument('--base-path', type=str, default=None, help='Base path for the project')
#     parser.add_argument('--ilastik-path', type=str, default=None, help='Path to the Ilastik executable')
#
#     args = parser.parse_args()
#     return args


def set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Create a console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)