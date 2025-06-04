# -*- coding: utf-8 -*-
#  Copyright 2024 Till Leissner
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#   Authored by: Till Leissner (University of Southern Denmark)
#   Edited by: 
#
#   We acknowledge support from the ESS Lighthouse on Hard Materials in 3D, SOLID, funded by the Danish Agency for Science and Higher Education (grant No. 8144-00002B).


### General imports 
import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

### Import modules for image processing
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import imageio

### Parallel computing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

### Import all CIL components needed
from cil.framework import AcquisitionGeometry


def set_geometry(logfile,infofile=None):
    """
    Creates a CIL AcquisitionGeometry object for a cone-beam CT scan.

    Args:
        infofile (str): Path to the information file.
        logfile (str): Path to the log file.

    Returns:
        AcquisitionGeometry: A geometry object representing the scan setup.
    """

    # Extract scan parameters from the log file
    scanparams = get_scanparams(logfile)

    # Get angles in degrees and radians
    if infofile:
        theta_deg, theta_rad = get_angles_from_infofile(infofile)
    else: 
        theta_deg, theta_rad = get_angles_from_logfile(scanparams)


    # Calculate distances
    distance_source_origin = float(scanparams['Acquisition']['object_to_source_(mm)'])
    distance_origin_detector = float(scanparams['Acquisition']['camera_to_source_(mm)'])-distance_source_origin
    
    # Get detector properties
    detector_pixel_size = float(scanparams['System']['camera_pixel_size_(um)'])/1000
    detector_rows = int(scanparams['Acquisition']['number_of_rows'])
    detector_cols = int(scanparams['Acquisition']['number_of_columns'])

    # Create AcquisitionGeometry object
    ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-distance_source_origin,0],detector_position=[0,distance_origin_detector,0],
                                       detector_direction_x=[1,0,0],detector_direction_y=[0,0,1]
                                      )\
    .set_panel(num_pixels=[detector_cols,detector_rows], pixel_size = detector_pixel_size)\
    .set_angles(angles=theta_deg)

    return ag




def get_angles_from_logfile(scanparams):
    """
    Calculates angles in degrees from the information in the log file.

    Args:
        logfile (str): Path to the log file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - First array: Angles in degrees.
            - Second array: Angles converted to radians.
    """
    angles_deg = [ i*float(scanparams['Acquisition']['rotation_step_(deg)']) for i in range(int(scanparams['Acquisition']['number_of_files'])) ]
    return np.array(angles_deg), np.deg2rad(angles_deg)


def get_angles_from_infofile(infofile, startangle=0):
    """
    Extracts angles in degrees from an information file.

    Args:
        infofile (str): Path to the information file.
        startangle (float, optional): Starting angle in degrees (default is 0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - First array: Angles in degrees.
            - Second array: Angles converted to radians.
    """
    
    # Initialize the list of angles with the start angle
    angles_deg = [startangle]
    
    # Read the information file
    with open(infofile, "r") as f:
        for line in f.readlines():
                line = line.rstrip('\n')
                if 'Drift compensation scan' in line:
                        break
                elif 'achieved angle' in line:
                        angles_deg.append(float(line.strip().rsplit(' ',1)[-1][:-1]))
    return np.array(angles_deg), np.deg2rad(angles_deg)


def get_reference_angles_from_infofile(infofile, startangle=0):
    """
    Extracts angles in degrees from an information file, starting after the
    "Drift compensation scan" line.

    Args:
        infofile (str): Path to the information file.
        startangle (float, optional): Starting angle in degrees (default is 0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - First array: Angles in degrees.
            - Second array: Angles converted to radians.
    """
    angles_deg = [startangle]
    start_collecting = False
    with open(infofile, "r") as f:
        for line in f:
            line = line.strip()
            if 'Drift compensation scan' in line:
                start_collecting = True
                continue
            if start_collecting and 'achieved angle' in line:
                angles_deg.append(float(line.strip().rsplit(' ',1)[-1][:-1]))
    return np.array(angles_deg), np.deg2rad(angles_deg)


def get_scanparams(logfile):
    """
    Extracts scan parameters from a log file.

    Args:
        logfile (str): Path to the log file.

    Returns:
        dict: A dictionary containing scan parameters organized by sections.
    """
    
    # Initialize an empty dictionary to store scan parameters
    scanparams = {}

    # Read the log file
    with open(logfile, "r") as f:
        for line in f.readlines():
                line = line.rstrip('\n')
                if line.startswith('['):
                        section = line[1:][:-1]
                        scanparams[section] ={}
                else:
                        scanparams[section][str.split(line, '=')[0].replace(' ','_').lower()]=str.split(line, '=')[1]
        return scanparams
        

def get_filelist(datadir,dataset_prefix,num_digits=8,extension='tif'):
    """
    Retrieves a sorted list of file paths matching a specific pattern.

    Args:
        datadir (str): Directory path where the files are located.
        dataset_prefix (str): Prefix for the dataset filenames.
        num_digits (int, optional): Number of digits in the numeric part of the filenames (default is 8).

    Returns:
        List[str]: A sorted list of file paths matching the specified pattern.
    """
    return sorted(glob.glob(datadir+'/'+dataset_prefix+('[0-9]' * num_digits)+'.'+extension))



def rename_files(directory, old_prefix, new_prefix):
    #Usage: bc.rename_files('./samples/02-switch-large', '001', '001_')
    for filename in os.listdir(directory):
        if filename.startswith(old_prefix):
            new_name = filename.replace(old_prefix, new_prefix, 1)
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))


### Thermal drift alignment functions
def shift_and_save_star(args):
    return shift_and_save(*args)

def shift_and_save(index, image, shift_x, shift_y, output_dir, output_prefix):
    shifted = shift(image, (shift_y, shift_x))
    filename = os.path.join(output_dir, f"{output_prefix}{index:04d}.tiff")
    imageio.imwrite(filename, shifted.astype(np.uint16))
    return index, shifted  # To reconstruct the new stack in order

def align_projection_stack(
    proj,                          # Projection stack object with .as_array() and .fill()
    infofile,                     # Path to infofile for angle extraction
    datadir,                      # Directory containing raw data files
    dataset_prefix,               # Prefix for raw dataset filenames
    filelist,                     # List of raw projection filenames (ordered)
    output_dir,                   # Directory to save shifted .tiff images
    output_prefix="aligned_",     # Prefix for saved .tiff images
    exact_match=False             # Match angles exactly or within half-step tolerance
):
    # --- Ensure output directory exists ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Load angles and filenames ---
    corr_angles_deg, _ = get_reference_angles_from_infofile(infofile, startangle=0)
    corr_filelist = get_filelist(datadir, dataset_prefix, num_digits=8, extension="iif")
    theta_deg, _ = get_angles_from_infofile(infofile)

    df_proj = pd.DataFrame({'filename': filelist, 'angle': theta_deg})
    df_corr = pd.DataFrame({'filename': corr_filelist, 'angle': corr_angles_deg})

    # --- Determine matching tolerance ---
    tolerance = 0 if exact_match else (df_proj["angle"].iloc[1] - df_proj["angle"].iloc[0]) / 2

    # --- Match projection and correction images by angle ---
    df_proj['key'] = df_corr['key'] = 1
    merged = pd.merge(df_proj, df_corr, on='key', suffixes=('_proj', '_corr')).drop('key', axis=1)
    matched = merged[np.abs(merged['angle_proj'] - merged['angle_corr']) <= tolerance].reset_index(drop=True)

    # --- Plot differences ---
    n = int(np.ceil(np.sqrt(len(matched))))
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    axes = axes.flatten()

    shifts_x, shifts_y = [], []
    for i, row in matched.iterrows():
        img_pre = imageio.v2.imread(row['filename_proj'])
        img_post = imageio.v2.imread(row['filename_corr'])

        diff = img_post.astype(float) - img_pre.astype(float)
        axes[i].imshow(diff, cmap='seismic', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        axes[i].axis('off')
        axes[i].set_title(f"{row['angle_proj']}°")

        shift_yx, error, _ = phase_cross_correlation(img_pre, img_post, upsample_factor=100)
        print(f"[{i}] Angle {row['angle_proj']}°: Drift (y, x) = {shift_yx}, error = {error}")

        shifts_y.append(shift_yx[0])
        shifts_x.append(shift_yx[1])

    for j in range(len(matched), len(axes)):
        axes[j].axis('off')

    fig.suptitle("Difference: Post - Pre scans")
    plt.tight_layout()
    plt.show()

    # --- Fit linear drift model ---
    A = np.vstack([corr_angles_deg, np.ones(len(corr_angles_deg))]).T
    mx, cx = np.linalg.lstsq(A, shifts_x, rcond=None)[0]
    my, cy = np.linalg.lstsq(A, shifts_y, rcond=None)[0]

    shifts_x_all = [mx * angle + cx for angle in theta_deg]
    shifts_y_all = [my * angle + cy for angle in theta_deg]

    # --- Plot fitted shift curves ---
    plt.scatter(corr_angles_deg, shifts_x, color='blue', label="x (measured)")
    plt.plot(theta_deg, shifts_x_all, color='blue', linestyle='--', label="x (fit)")
    plt.scatter(corr_angles_deg, shifts_y, color='orange', label="y (measured)")
    plt.plot(theta_deg, shifts_y_all, color='orange', linestyle='--', label="y (fit)")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Shift (pixels)")
    plt.legend()
    plt.title("Fitted drift correction over angle")
    plt.grid(True)
    plt.show()

    # --- Apply shifts to image stack ---
    print(f"Start image alignment using {os.cpu_count()} workers")
    img_stack = proj.as_array()

    # Prepare arguments
    args = [
        (i, img_stack[i], shifts_x_all[i], shifts_y_all[i], output_dir, output_prefix)
        for i in range(len(img_stack))
    ]

    # Run in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(shift_and_save_star, args))

    # Sort and rebuild the image stack
    results.sort()  # sort by index
    img_stack_shifted = np.stack([res[1] for res in results])

    print(f"Saved aligned image stack to '{output_dir}' with prefix '{output_prefix}'.")
    
    return {
        "img_stack": img_stack,
        "shifts_x": shifts_x_all,
        "shifts_y": shifts_y_all,
        "fit_x": (mx, cx),
        "fit_y": (my, cy)
    }
