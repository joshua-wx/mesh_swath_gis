import numpy as np
import pyart
import wradlib.clutter as clutter
import time

def _filter_hardcoding(my_array, nuke_filter, bad=-9999):
    """
    Harcoding GateFilter into an array.
    Parameters:
    ===========
        my_array: array
            Array we want to clean out.
        nuke_filter: gatefilter
            Filter we want to apply to the data.
        bad: float
            Fill value.
    Returns:
    ========
        to_return: masked array
            Same as my_array but with all data corresponding to a gate filter
            excluded.
    """
    filt_array = np.ma.masked_where(nuke_filter.gate_excluded, my_array.copy())
    filt_array = filt_array.filled(fill_value=bad)
    to_return = np.ma.masked_where(filt_array == bad, filt_array)
    return to_return    

def main(radar, tilt_list, ref_fieldname):
    """
    Apply a clutter removal workflow to a single radar volume
    ===============
    (1) Gabella texture filter
    (2) Despeckle
    Parameters:
    ===============
        radar: pyart radar object
        
        tilt_list: list
            tilt list to process. Empty triggers processing for all tilts
    Returns:
    ===============
        radar: pyart radar object
    """

    #config
    rain_cut_dbz  = 10.
    #min_dbz       = 0.
    #snr_threshold = 10
    
    #define the indices for the required sweep
    sweep_startidx = radar.sweep_start_ray_index['data'][:]
    sweep_endidx   = radar.sweep_end_ray_index['data'][:]
    refl_data      = radar.fields[ref_fieldname]['data'].copy()
    clutter_mask   = np.zeros_like(refl_data) #clutter flag = 1, no clutter = 0

    #build list of tilts to process for gabella filter
    if not tilt_list:
        tilt_list = np.arange(len(sweep_startidx))
    else:
        tilt_list = np.array(tilt_list)
        
    #loop through sweeps    
    for k in tilt_list:
        #extract ppi
        ppi            = refl_data[sweep_startidx[k]:sweep_endidx[k]+1]
        #generate clutter mask for ppi
        clmap = clutter.filter_gabella(ppi,
                                       wsize=5,
                                       thrsnorain=rain_cut_dbz,
                                       tr1=10.,
                                       n_p=8,
                                       tr2=1.3)
        #insert clutter mask for ppi into volume mask
        clutter_mask[sweep_startidx[k]:sweep_endidx[k]+1] = clmap

    #add clutter mask to radar object
    clutter_field = {'data': clutter_mask, 'units': 'mask', 'long_name': 'Gabella clutter mask',
                            'standard_name': 'CLUTTER', 'comments': 'wradlib implementation of Gabella et al., 2002'}
    radar.add_field('reflectivity_clutter', clutter_field, replace_existing=True) 
    
    #apply clutter filter to gatefiler
    gatefilter = pyart.correct.GateFilter(radar)
    gatefilter.exclude_equal('reflectivity_clutter',1)
    
    #generate depseckle filter
    gate_range       = radar.range['meters_between_gates']
    #apply limits
    if gate_range < 250:
        print('gate range too small in clutter.py, setting to 250m')
        gate_range = 250
    if gate_range > 1000:
        print('gate range too large in clutter.py, setting to 1000m')
        gate_range = 1000
    #rescale and calculate sz
    despeckle_factor = 1000/gate_range #scale despeckle according to gate size 
    despeckle_sz     = 15 * despeckle_factor #1000m = 15, 500m = 30, 250m = 60
    #apply despeckle to gatefilter
    gatefilter       = pyart.correct.despeckle.despeckle_field(radar, ref_fieldname, gatefilter=gatefilter, size=despeckle_sz)
    
    #apply filter to mask
    cor_refl_data = np.ma.masked_where(gatefilter.gate_excluded, refl_data.copy())

    #return radar object
    return cor_refl_data, gatefilter