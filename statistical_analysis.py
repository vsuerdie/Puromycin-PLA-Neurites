import os
import re
import pandas as pd
import numpy as np
import tifffile as tiff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Tuple
import logging
from utils import set_logger, get_args


def create_output_directories(result_path: str) -> Dict[str, str]:
    directories =  {name: os.makedirs(path := os.path.join(result_path, name), exist_ok=True) or path
                    for name in ['result', 'plots', 'boxplots']}
    return directories


def load_and_match_files(masks_folder: str, csv_folder: str) -> Tuple[List[str], List[str]]:
    mask_files = [f for f in os.listdir(masks_folder) if f.endswith('filaments.tif')]
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    return mask_files, csv_files


def process_single_image(mask_path: str, csv_path: str, resolution: float = 0.063) -> Dict:
    mask = tiff.imread(mask_path)
    data = pd.read_csv(csv_path)
    # Convert coordinates to integers and filter valid positions
    x, y = data['x'].astype(int), data['y'].astype(int)
    max_y, max_x = mask.shape
    data = data[(x >= 0) & (x < max_x) & (y >= 0) & (y < max_y)]
    # Filter signals on filaments
    filtered_data = data[mask[y, x] == 2]
    num_signals = filtered_data.shape[0]
    # Calculate areas
    mask_pixel_area = (mask == 2).sum()
    mask_area_micrometers = round(mask_pixel_area * (resolution ** 2))
    counts_per_area = round((num_signals / mask_area_micrometers) if mask_area_micrometers > 0 else 0, 4)

    return {
        'filtered_data': filtered_data,
        'num_signals': num_signals,
        'mask_area_micrometers': mask_area_micrometers,
        'counts_per_area': counts_per_area,
        'mask': mask
    }

def create_and_save_spots_over_mask_plots(mask: np.ndarray, filtered_data: pd.DataFrame, mask_key: str, output_path: str):
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.scatter(filtered_data['x'], filtered_data['y'], color='red', marker='o', s=1)
    plt.gca().invert_yaxis()
    plt.title(f'Filtered Coordinates for {mask_key}')
    plt.savefig(output_path)
    plt.close()


def add_significance_annotation(ax: plt.Axes, x: str, y: str, data: pd.DataFrame, p_value: float = None):
    if p_value is None:
        groups = [data[y][data[x] == label] for label in data[x].unique()]
        _, p_value = stats.ttest_ind(*groups)

    significance = '****' if p_value < 0.0001 else '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    x_positions = [i for i, label in enumerate(data[x].unique())]
    x_min, x_max = min(x_positions), max(x_positions)
    y_max = data[y].max() + (data[y].max() * 0.1)

    ax.plot([x_min, x_min, x_max, x_max], [y_max, y_max + 0.025, y_max + 0.025, y_max], color='black', lw=1.5)
    ax.text((x_min + x_max) / 2, y_max + 0.025, significance, ha='center', va='bottom', color='black', fontsize=12)


def create_and_save_boxplot(df: pd.DataFrame, y_variable: str, ylabel: str, title: str, output_path: str):
    fig, ax = plt.subplots(figsize=(4.5, 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sns.boxplot(x='sample_type', hue ='sample_type', y=y_variable, data=df, palette="muted", ax=ax, width=0.6, legend=False)
    sns.stripplot(x='sample_type', hue ='sample_type', y=y_variable, data=df, ax=ax, jitter=True, dodge=False, linewidth=1, palette='dark', legend=False)

    plt.ylabel(ylabel, fontsize=15)
    plt.title(title)
    plt.ylim(0, df[y_variable].max() + df[y_variable].max() * 0.5)
    plt.xticks(fontsize=10)

    add_significance_annotation(ax, 'sample_type', y_variable, df)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def process_all_images(input_masks_path: str, input_csv_path: str, directories: Dict[str, str]) -> List[Dict]:
    results = []
    for mask_file in os.listdir(input_masks_path):
        if mask_file.endswith('filaments.tif'):
            mask_key = re.search(r'(\d+-\d+)', mask_file).group(1)
            matching_csvs = [f for f in os.listdir(input_csv_path) if re.search(rf'{mask_key}(?!\d)', f)]

            if len(matching_csvs) != 1:
                logging.warning('......')
            else:
                mask_path = os.path.join(input_masks_path, mask_file)
                csv_path = os.path.join(input_csv_path, matching_csvs[0])

                analysis_results = process_single_image(mask_path, csv_path)

                # Save rs-fish spots on mask plot
                plot_path = os.path.join(directories['plots'], f'{mask_key}_plot.png')
                create_and_save_spots_over_mask_plots(analysis_results['mask'], analysis_results['filtered_data'], mask_key, plot_path)

                # Add results to list
                results.append({
                    'sample_type': mask_key.split('-')[0],
                    'sample': mask_key,
                    'area_of_filaments': analysis_results['mask_area_micrometers'],
                    'total_counts': analysis_results['num_signals'],
                    'counts_per_area': analysis_results['counts_per_area']
                })
    return results


def save_results_df(result_folder: str, results: List[Dict]):
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(result_folder, 'filament_analysis_results.csv'), index=False)

    return results_df


def create_and_save_all_boxplots(result_path: str, directories: Dict[str, str], results_df: pd.DataFrame):
    boxplot_configs = [
        ('counts_per_area', 'Laminin-Puro-PLA per area', 'Translation Events in Neurites', 'smFISH_per_area_Laminin.png'),
        ('total_counts', 'Total Counts', 'Total Counts by Sample Type', 'total_counts_by_sample_type.png'),
        ('area_of_filaments', 'Area of Filaments', 'Area of Filaments by Sample Type', 'area_of_filaments_by_sample_type.png')
    ]

    for y_var, ylabel, title, filename in boxplot_configs:
        output_path = os.path.join(directories['boxplots'], filename)
        create_and_save_boxplot(results_df, y_var, ylabel, title, output_path)


def main():
    set_logger()
    logging.info('Logger set')

    arg_names = ['analysis_base_path']
    args = get_args(arg_names)
    base_path = args['analysis_base_path']

    input_masks_path = os.path.join(base_path, "masks")
    input_csv_path = os.path.join(base_path, "RS_FISH/RS_FISH_Results")
    result_path = os.path.join(base_path, "statistical_analysis")

    directories = create_output_directories(result_path)

    results = process_all_images(input_masks_path, input_csv_path, directories)
    result_df = save_results_df(result_path, results)
    create_and_save_all_boxplots(result_path, directories, result_df)


if __name__ == "__main__":
    main()