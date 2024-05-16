# General imports 
import numpy as np
import glob

### Import all CIL components needed
from cil.framework import AcquisitionGeometry

def set_geometry(infofile,logfile):
    theta_deg, theta_rad = get_angles_deg(infofile)
    scanparams = get_scanparams(logfile)
    distance_source_origin = float(scanparams['Acquisition']['object_to_source_(mm)'])
    distance_origin_detector = float(scanparams['Acquisition']['camera_to_source_(mm)'])-distance_source_origin
    detector_pixel_size = float(scanparams['System']['camera_pixel_size_(um)'])/1000
    detector_rows = int(scanparams['Acquisition']['number_of_rows'])
    detector_cols = int(scanparams['Acquisition']['number_of_columns'])

    ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-distance_source_origin,0],detector_position=[0,distance_origin_detector,0],
                                       detector_direction_x=[1,0,0],detector_direction_y=[0,0,1]
                                      )\
    .set_panel(num_pixels=[detector_cols,detector_rows], pixel_size = detector_pixel_size)\
    .set_angles(angles=theta_deg)

    return ag

def get_angles_deg(infofile, startangle=0):
        angles_deg = [startangle]
        toggle = True

        f = open(infofile, "r")
        for line in f.readlines():
                line = line.rstrip('\n')
                if 'Drift compensation scan' in line:
                        break
                elif 'achieved angle' in line:
                        angles_deg.append(float(line.strip().rsplit(' ',1)[-1][:-1]))
        return np.array(angles_deg), np.deg2rad(angles_deg)

def get_scanparams(logfile):
        scanparams = {}
        f = open(logfile, "r")
        for line in f.readlines():
                line = line.rstrip('\n')
                if line.startswith('['):
                        section = line[1:][:-1]
                        scanparams[section] ={}
                else:
                        scanparams[section][str.split(line, '=')[0].replace(' ','_').lower()]=str.split(line, '=')[1]
        return scanparams

def get_filelist(datadir,dataset_prefix,num_digits=8):
    return sorted(glob.glob(datadir+'/'+dataset_prefix+('[0-9]' * num_digits)+'.tif'))
