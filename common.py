import os
from datetime import datetime
import tempfile
import zipfile

import netCDF4
import cftime
import numpy as np
import pandas as pd
from osgeo import gdal
from pysteps import motion
from scipy.ndimage import map_coordinates

from matplotlib import pyplot as plt

import mesh
import clutter

import pyart

def unpack_zip(zip_ffn):
    """
    Unpacks zip file in temp directory

    Parameters:
    ===========
        zip_ffn: str
            Full filename to zip file 
            
    Returns:
    ========
        temp_dir: string
            Path to temp directory
    """    
    #build temp dir
    temp_dir = tempfile.mkdtemp()
    #unpack tar
    zip_fd = zipfile.ZipFile(zip_ffn)
    zip_fd.extractall(path=temp_dir)
    zip_fd.close()
    return temp_dir

def get_dt_list(vol_ffn_list):
    dt_list = []
    for vol_ffn in vol_ffn_list:
        dt_list.append(datetime.strptime(os.path.basename(vol_ffn)[3:18],'%Y%m%d_%H%M%S'))
    return dt_list

def get_projparams(vol_ffn):
    #read radar data
    radar = pyart.aux_io.read_odim_h5(vol_ffn, file_field_names=True)
    #read location
    site_lon = radar.longitude['data']
    site_lat = radar.latitude['data']
    #parse projparams
    projparams = {'proj':'pyart_aeqd',
                 'lon_0':site_lon,
                 'lat_0':site_lat}
    return projparams

def mkdir(mydir):
    if os.path.exists(mydir):
        return None
    try:
        os.makedirs(mydir)
    except FileExistsError:
        return None
    return None

def vol_to_grids(vol_ffn, grid_config, vol_dbz_offset):
    #refl cappi index
    #5 = 2.5km
    cappi_index = 5
    #read radar
    radar = pyart.aux_io.read_odim_h5(vol_ffn, file_field_names=True)
    try:
        #check if DBZH_CLEAN is present
        refl_field = radar.fields['DBZH_CLEAN']['data']
    except:
        #run clutter filtering if DBZH_CLEAN is missing
        refl_field, _ = clutter.main(radar, [], 'DBZH')
    #run range filtering
    R, A = np.meshgrid(radar.range['data']/1000, radar.azimuth['data'])
    ma_refl_field = np.ma.masked_where(np.logical_or(R<10, R>120), refl_field.copy())
    radar.add_field_like('DBZH', 'MA_DBZH_CLEAN', ma_refl_field, replace_existing=True)

    #temp data <-- get from ACCESS-R archive
    temp_data = mesh.temperature_profile_access(radar)

    #grid
    grid = pyart.map.grid_from_radars(
                    radar,
                    grid_shape = grid_config['GRID_SHAPE'],
                    grid_limits = grid_config['GRID_LIMITS'],
                    roi_func = 'constant',
                    constant_roi = grid_config['GRID_ROI'],
                    weighting_function = 'Barnes2',
                    fields = ['MA_DBZH_CLEAN'])
    #extract data and apply refl offset
    refl_grid = grid.fields['MA_DBZH_CLEAN']['data'] - vol_dbz_offset
    alt_vec = grid.z['data']
    #calculate refl field
    REFL = refl_grid[cappi_index,:,:]
    REFL_meta = {'units': 'dBZ', 'long_name': 'Horizontal Reflectivity Factor',
                'standard_name': 'reflectivity', 'comments': 'calibrated reflectivity'}
    #convert to mesh
    MESH, MESH_meta = mesh.main(refl_grid, alt_vec, temp_data)
    
#     fig = plt.figure(figsize=[10,10])
#     plt.imshow(REFL)
    
    #return 2D mesh grid
    return MESH, MESH_meta, REFL, REFL_meta

def write_netcdf(flist_dt, level_2_path, grid_config, field_data, field_name, field_meta, gnrl_meta, is_file, projparams, fillvalue=-32768):
    
    """
    Generates output netcdf files
    """
    #create dims
    grid_range = grid_config['GRID_LIMITS'][1][1]
    grid_step = grid_config['GRID_STEP']
    x = np.arange(-grid_range, grid_range+grid_step, grid_step)
    y = x.copy()
    lon, lat = pyart.core.transforms.cartesian_vectors_to_geographic(x, y, projparams)
    
    # generate output filename
    field_str = field_name.upper()
    out_dir   = level_2_path
    mkdir(out_dir)
    out_fn    = '_'.join([grid_config['radar_id_str'], flist_dt[0].strftime('%Y%m%d'), field_str]) + '.nc'
    out_ffn   = '/'.join([out_dir, out_fn])
    if os.path.exists(out_ffn):
        print(f'Output file {out_ffn} already exists. Removing old file')
        cmd = 'rm -rf ' + out_ffn
        os.system(cmd)
    
    # Generate netcdf time dimension
    time_unit = f'seconds since {str(flist_dt[0])}'
    time = netCDF4.date2num(flist_dt, time_unit).astype(np.int32)
    T_DIM = len(time)
    X_DIM = len(x)
    Y_DIM = len(y)
    
    #mask data
    field_data = np.ma.masked_where(np.isnan(field_data), field_data)
    
    
    # Write data
    with netCDF4.Dataset(out_ffn, 'w') as ncid:
        ncid.createDimension('time', T_DIM)
        ncid.createDimension("x", X_DIM)
        ncid.createDimension("y", Y_DIM)

        myfield = ncid.createVariable(field_name, np.float32, ("time", "y", "x"), 
                                       zlib=True, fill_value=fillvalue, least_significant_digit=2)
        ncquality = ncid.createVariable('isfile', is_file.dtype, ("time",))
        nctime = ncid.createVariable('time', time.dtype, 'time')
        
        nclon = ncid.createVariable('longitude', np.float32, ('y', 'x'), zlib=True, least_significant_digit=2)
        nclat = ncid.createVariable('latitude', np.float32, ('y', 'x'), zlib=True, least_significant_digit=2)
        nclon[:] = lon
        nclon.units = 'degrees_east'
        nclat[:] = lat
        nclat.units = 'degrees_north'
        
        ncx = ncid.createVariable('x', np.int32, ('x'), zlib=True)
        ncy = ncid.createVariable('y', np.int32, ('y'), zlib=True)
        ncx[:] = x
        ncx.units = 'm'
        ncy[:] = y
        ncy.units = 'm'
        
        nctime[:] = time
        nctime.units = time_unit
        ncquality[:] = is_file
        ncquality.units = ''
        ncquality.setncattr('description', "0: no data, 1: data available at this time step")
        
        # Write attributes
        myfield[:] = field_data.filled(fillvalue)
        for k, v in field_meta.items():
            if k == '_FillValue':
                continue
            try:
                myfield.setncattr(str(k), str(v))
            except AttributeError:
                raise

        for k, v  in gnrl_meta.items():
            ncid.setncattr(k, str(v))

    return out_ffn

def write_grid_geotiff(input_array, filename, geo_info,
                       vmin=0, vmax=75):
    """
    Write a 2D array to a GeoTIFF file.
    
    Parameters
    ----------
    input_array : numpy array
        Grid to write to file.
    filename : str
        Filename for the GeoTIFF.
    geo_info : dict
        contains grid_step, grid_dist, radar_lat, radar_lon
        
    Other Parameters
    ----------------
    vmin : int or float, optional
        Minimum value to color for RGB output or SLD file.
    vmax : int or float, optional
        Maximum value to color for RGB output or SLD file.
    """
    
    #create temp file
    temp_fn = '/tmp/tmp_geotif.tif'
    
    dist      = geo_info['grid_dist']
    rangestep = geo_info['grid_step']
    lat       = geo_info['radar_lat'] #lat origin
    lon       = geo_info['radar_lon'] #lon origin (middle of grid)
    
    # Check if masked array; if so, fill missing data
    data = input_array.astype(float)

    iproj = 'PROJCS["unnamed",GEOGCS["WGS 84",DATUM["unknown",' + \
        'SPHEROID["WGS84",6378137,298.257223563]],' + \
        'PRIMEM["Greenwich",0],' + \
        'UNIT["degree",0.0174532925199433]],' + \
        'PROJECTION["Azimuthal_Equidistant"],' + \
        'PARAMETER["latitude_of_center",' + str(lat) + '],' + \
        'PARAMETER["longitude_of_center",' + str(lon) + '],' + \
        'PARAMETER["false_easting",0],' + \
        'PARAMETER["false_northing",0],' + \
        'UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'
    out_driver = gdal.GetDriverByName("GTiff")

    # Single-channel, floating-point output
    dst_options = ['COMPRESS=LZW', 'ALPHA=YES']
    dst_ds = out_driver.Create(
        temp_fn, data.shape[1], data.shape[0], 1,
        gdal.GDT_Float32, dst_options)

    # Common Projection and GeoTransform
    dst_ds.SetGeoTransform([-dist, rangestep, 0, dist, 0, -rangestep])
    dst_ds.SetProjection(iproj)

    # Final output
    dst_ds.GetRasterBand(1).WriteArray(data[::-1, :])
    dst_ds.FlushCache()
    dst_ds = None

    #convert to EPSG3577 GDA
    input_raster = gdal.Open(temp_fn)
    gdal.Warp(filename, input_raster, dstSRS="EPSG:3577")
    
    os.system('rm '+ temp_fn)
    
    
def advection_correction(R, T=5, t=1):
    """
    R = np.array([qpe_previous, qpe_current])
    T = time between two observations (5 min)
    t = interpolation timestep (1 min)
    """

    # Evaluate advection
    oflow_method = motion.get_method("LK")
    fd_kwargs = {"buffer_mask": 10}  # avoid edge effects
    V = oflow_method(R, fd_kwargs=fd_kwargs)

    # Perform temporal interpolation
    Rd = np.zeros((R[0].shape))
    x, y = np.meshgrid(
        np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float),
    )
    for i in range(t, T + t, t):

        pos1 = (y - i / T * V[1], x - i / T * V[0])
        R1 = map_coordinates(R[0], pos1, order=1)

        pos2 = (y + (T - i) / T * V[1], x + (T - i) / T * V[0])
        R2 = map_coordinates(R[1], pos2, order=1)

        Rd_temp = np.amax(np.stack((R1, R2), axis=2), axis=2)
        
        Rd = np.amax(np.stack((Rd, Rd_temp), axis=2), axis=2)

    return Rd