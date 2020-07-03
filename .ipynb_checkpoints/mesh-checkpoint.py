"""
MESH sub-module of pyhail

Contains the single pol MESH retrieval for gridded radar data.
Required reflectivity and temperature data

Joshua Soderholm - 15 June 2018
"""

import pyart
import netCDF4
import numpy as np

def _sounding_interp(snd_temp, snd_height, target_temp):
    """
    Provides an linear interpolated height for a target temperature using a
    sounding vertical profile. Looks for first instance of temperature
    below target_temp from surface upward.

    Parameters
    ----------
    snd_temp : ndarray
        Temperature data (degrees C).
    snd_height : ndarray
        Relative height data (m).
    target_temp : float
        Target temperature to find height at (m).

    Returns
    -------
    intp_h: float
        Interpolated height of target_temp (m).

    """
    intp_h = np.nan

    # find index above and below freezing level
    mask = np.where(snd_temp < target_temp)
    above_ind = mask[0][0]
    # check to ensure operation is possible
    if above_ind > 0:
        # index below
        below_ind = above_ind - 1
        # apply linear interplation to points above and below target_temp
        set_interp = interp1d(
            snd_temp[below_ind:above_ind+1],
            snd_height[below_ind:above_ind+1], kind='linear')
        # apply interpolant
        intp_h = set_interp(target_temp)
        return intp_h
    else:
        return target_temp[0]

def _get_latlon(radgrid, ref_name):
    """
    Generates lattitude and longitude arrays.
    Parameters:
    ===========
    radgrid: struct
        Py-ART grid object.
    Returns:
    ========
    longitude: ndarray
        Array of coordinates for all points.
    latitude: ndarray
        Array of coordinates for all points.
	
	From cpol_processing: https://github.com/vlouf/cpol_processing
    """
    # Declare array, filled 0 in order to not have a masked array.
    lontot = np.zeros_like(radgrid.fields[ref_name]['data'].filled(0))
    lattot = np.zeros_like(radgrid.fields[ref_name]['data'].filled(0))

    for lvl in range(radgrid.nz):
        lontot[lvl, :, :], lattot[lvl, :, :] = radgrid.get_point_longitude_latitude(lvl)

    longitude = pyart.config.get_metadata('longitude')
    latitude  = pyart.config.get_metadata('latitude')

    longitude['data'] = lontot
    latitude['data']  = lattot

    return longitude, latitude

def main(refl_grid, alt_vec, temph_data):

    """
 	Hail grids adapted from Witt et al. 1998 and 95th percentile fit from Muillo and Homeyer 2019
    Exapnded to grids (adapted from wdss-ii)

    Parameters:
    ===========
    refl_grid: 3D array
        reflectivity array
    alt_vec: 1D alt_vec
        temperatue profile at radar site
	temph_data: list
		contains 0C and -20C altitude (m) in first and second element position, only used if snd_input is empty
    Returns:
    ========
    None, write to file
	
    """

    #MESH constants
    z_lower_bound = 40
    z_upper_bound = 50
    
    #extract temp data
    snd_t_0C       = temph_data[0]
    snd_t_minus20C = temph_data[1]
            
    #tilt altitude in 3D
    grid_sz   = np.shape(refl_grid)
    alt_grid  = np.tile(alt_vec,(grid_sz[1], grid_sz[2], 1))
    alt_grid  = np.swapaxes(alt_grid, 0, 2) #m
    
    #calc reflectivity weighting function
    weight_ref                             = (refl_grid - z_lower_bound)/(z_upper_bound - z_lower_bound)
    weight_ref[refl_grid <= z_lower_bound] = 0
    weight_ref[refl_grid >= z_upper_bound] = 1
    
    #calc hail kenitic energy
    hail_KE = (5 * 10**-6) * 10**(0.084 * refl_grid) * weight_ref
    
    #calc temperature based weighting function
    weight_height = (alt_grid - snd_t_0C) / (snd_t_minus20C - snd_t_0C)
    weight_height[alt_grid <= snd_t_0C]       = 0
    weight_height[alt_grid >= snd_t_minus20C] = 1

    #calc severe hail index
    grid_sz_m = alt_vec[1] - alt_vec[0]
    SHI = 0.1 * np.sum(weight_height * hail_KE, axis=0) * grid_sz_m

    #calc maximum estimated severe hail (mm)
    MESH = 17.270 * SHI**0.272
    
    MESH_meta        = {'units': 'mm', 'long_name': 'Maximum Expected Size of Hail',
                        'standard_name': 'MESH', 'comments': '95th percentile fit from Muillo and Homeyer 2019, only valid in the first level'}

    return MESH, MESH_meta