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
#

### Import all CIL components needed
#

### Functions

def set_geometry(logfile,infofile=None):
    from cil.framework import AcquisitionGeometry
    from .import_utils import get_angles_from_logfile, get_angles_from_infofile, get_scanparams
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





