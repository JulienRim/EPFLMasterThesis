import tdt
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy import signal, integrate, stats
from scipy import interpolate
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from itertools import product

from GP_functions import *

###############################
def get_delta_heatmap(df, column_name = 'delta_rate', sort=True):
    """ Get the mean of the change in threshold crossing rates based on the electrode number and the stimulation channel.
        This creates a heatmap sorted to give most inihibitory to most excitatory (or not)
    
    Input
    -----
    df : pandas dataframe
        Dataframe from the output of MUA_stream_extract()
        It contains threshold crossings from a given block and stream. 
        
        
    column_name : string
        Name of the column to use as the values to create the heatmap
        Defaults to 'rates_diff' as that is the one that I use the most
    
    sort : bool
        A boolean paramater to determine whether or not the heatmap should be sorted from most inhibitory to most
        excitatory.
        False/0: do not sort
        True /1: sort 
    
    Output
    -------
    heatmap : pandas pivot table (technically also a dataframe)
        Sorted pivot table that has the average change in threshold crossings per stimulation channel and recording electrode
    """
    # Group by stim channel and electrode number
    heatmap = df.groupby(['recording_electrode', 'stimulation_electrode'])[[column_name]].mean().reset_index() \
          .pivot(index='recording_electrode', columns = 'stimulation_electrode', values=column_name)
    
    if sort:
        # Sorting by most inhibitory to most excitatory
        heatmap = heatmap.reindex(heatmap.T.mean().sort_values(ascending=False).index, axis=0)
        heatmap = heatmap.reindex(heatmap.mean().sort_values().index, axis=1)
    
    return heatmap

def fma_rec_spatial(delta_heatmap, stim_channel):
    """ Function to plot the changes in threshold crossings based on the spatial arangement of the electodes
        (for FMA) for recording channels
    
    Inputs 
    ------
    delta_heatmap : pandas pivot table
        Sorted pivot table that has the average change in threshold crossings per stimulation channel and recording electrode
        
    stim_channel : int
        The stimulation channel used to plot the spatial arangements 
        
        Suggested inputs:
            delta_heatmap[stim_channel].name    - specific channel
            delta_heatmap.columns[0]               - most inhibitory channel
                                                     since they are sorted, 0 is most inhibitory, 1 is the next most inhibitory etc.
            delta_heatmap.columns[-1]              - most excitatory channel
                                                     since they are sorted, -1 is most excitatory, -2 is the next most excitatory etc. 
    
    Outputs
    -------
    spatial_array : numpy array
        spatial array with the number of threshold crossings for each corresponding electrode
    """
    
    # Get the data and change column name to avoid errors
    thr_delta = pd.DataFrame(delta_heatmap[stim_channel]).reset_index()
    thr_delta = thr_delta.rename(columns={thr_delta.columns[1] : 'delta'})
    
    # This is the map of electrode locations based on the documentation for Floating Micro Arrays
    # The idea is that each electrode number here will be replaced by the number of threshold crossings for that given electrode
    A = np.array([[28, np.nan, 10, np.nan],
     [28, np.nan, 10, np.nan],
     [29, np.nan, 11, np.nan],
     [29,     20, 11,      2],
     [30,     20, 12,      2],
     [30,     21, 12,      3],
     [31,     21, 13,      3],
     [31,     22, 13,      4],
     [32,     22, 14,      4],
     [32,     23, 14,      5],
     [33,     23, 15,      5],
     [33,     24, 15,      6],
     [34,     24, 16,      6],
     [34,     25, 16,      7],
     [35,     25, 17,      7],
     [35,     26, 17,      8],
     [np.nan, 26, np.nan,  8],
     [np.nan, 27, np.nan,  9],
     [np.nan, 27, np.nan,  9]])
    
    # Create a dictionary to replace the values
    keys = thr_delta['recording_electrode']
    values = thr_delta['delta']
    d = dict(zip(keys, values))
    
    # The replacement
    spatial_array = np.copy(A)
    for electrode, metric in d.items():
        spatial_array[A == electrode] = metric
    
    # np.repeat to get plot to be wider else delete that line and uncomment the next one
    # return spatial_array
    return np.repeat(spatial_array, 2, axis=1)


def fma_stim_spatial(delta_heatmap, rec_channel):
    """ Function to plot the changes in threshold crossings based on the spatial arangement of the electodes
        (for FMA) for stimulation channels
    
    Inputs 
    ------
    delta_heatmap : pandas pivot table
        Sorted pivot table that has the average change in threshold crossings per stimulation channel and recording electrode
        
    rec_channel : int
        The recording channel used to plot the spatial arangements of stimulations
        Suggested inputs:
            delta_heatmap.loc[rec_channel].name    - specific recording channel (will give error if it is not available)
            delta_heatmap.iloc[rec_channel].name   - recording channel by index of the heatmap
                                                            e.g. delta_heatmap.iloc[0] gives the top channel
                                                                 delta_heatmap.iloc[31] gives the bottom channel
            delta_heatmap.sample().index           - random recording channel
            
    s_config : string
        Basically a way to determine if the plot will be made on the Utah array or the FMA array
    
    Outputs
    -------
    spatial_array : numpy array
        spatial array with the number of threshold crossings for each corresponding electrode
    """
    
    # Get the data and change column name to avoid errors
    thr_delta = pd.DataFrame(delta_heatmap.loc[rec_channel]).reset_index()
    thr_delta = thr_delta.rename(columns={thr_delta.columns[1] : 'delta'})
    
    # This is the map of electrode locations based on the documentation for Floating Micro Arrays
    # The idea is that each electrode number here will be replaced by the number of threshold crossings for that given electrode
    

    A = np.array([[28, np.nan, 10, np.nan],
     [28, np.nan, 10, np.nan],
     [29, np.nan, 11, np.nan],
     [29,     20, 11,      2],
     [30,     20, 12,      2],
     [30,     21, 12,      3],
     [31,     21, 13,      3],
     [31,     22, 13,      4],
     [32,     22, 14,      4],
     [32,     23, 14,      5],
     [33,     23, 15,      5],
     [33,     24, 15,      6],
     [34,     24, 16,      6],
     [34,     25, 16,      7],
     [35,     25, 17,      7],
     [35,     26, 17,      8],
     [np.nan, 26, np.nan,  8],
     [np.nan, 27, np.nan,  9],
     [np.nan, 27, np.nan,  9]])
    
    
    # Create a dictionary to replace the values
    keys = thr_delta['stimulation_electrode']
    values = thr_delta['delta']
    d = dict(zip(keys, values))
    
    # The replacement
    spatial_array = np.copy(A)
    for electrode, metric in d.items():
        spatial_array[A == electrode] = metric
    
    
    # np.repeat to get plot to be wider else delete that line and uncomment the next one
    # return spatial_array
    return np.repeat(spatial_array, 2, axis=1)



def get_electrode_maps(stream, block_number, verbose=True):
    """ This function takes a given stream and block number and returns the mapping of the electrodes to the 
        recording channel seen in the TDT block. 
    
    Inputs
    ------
    stream : TDT data stream
        e.g. data.streams.Wave ; data.streams.Wav2
        
    block_number : int
        The block number of the TDT data you are using
        
    verbose : bool
        Bool parameter to print what is happening. 
        Default is true
        
    
    Returns
    stim_electrode_map : pandas dataframe
        Map of the stimulation electrodes to TDT stimulation channels
        
    rec_electrode_map : pandas dataframe
        Map of the recroding electrodes to TDT recording channels
        
    s_config : str
        currently testing to see if this works
    r_config : str
        currently testing to see if this works
    """
    
    # load the map
    block_bank_map = pd.read_excel('block_bank_map.xlsx')
    experiment = block_bank_map.query('Block_number == {}'.format(block_number)).loc[block_bank_map.Wave == stream.name].reset_index()

    # Get the relevant bank numbers
    stimulation_bank = experiment.Stimulation_bank.values[0]
    recording_bank = experiment.Recording_bank.values[0]
    
    
    # Get the recording and stimulation locations (for plotting later)
    stimulation_location = experiment.Stimulation_location[0]
    recording_location = experiment.Recording_location[0]
    
    # Load the correct stimulation electrode map
    if stimulation_bank in [1, 2, 3, 7]:
        col_names = ['electrode_number', 'pin_number', 'TDT_channel']
        stim_electrode_map = pd.read_excel('FMA_electrode_map_arrays_1235.xlsx', usecols='A:C', names=col_names)
        stim_electrode_map = stim_electrode_map.astype({'TDT_channel' : 'float'})
        s_config = 'fma'
        if verbose:
            print('Stimulation Map is in FMA 1 2 3 5')
        
    elif stimulation_bank in [8, 9]:
        col_names = ['electrode_number', 'pin_number', 'TDT_channel']
        stim_electrode_map = pd.read_excel('FMA_electrode_map_arrays_67.xlsx', usecols='A:C', names=col_names)
        stim_electrode_map = stim_electrode_map.astype({'TDT_channel' : 'float'})
        s_config = 'fma'
        if verbose:
            print('Stimulation Map is in FMA 6 7')
            
    else:
        col_names = ['electrode_number', 'pin_number', 'recorded_channel', 'bank_number', 'TDT_channel']
        stim_electrode_map = pd.read_excel('Utah_electrode_map_array_4.xlsx', usecols='A:E', names=col_names)
        stim_electrode_map = stim_electrode_map.query('bank_number  == {}'.format(stimulation_bank)).reset_index()
        stim_electrode_map = stim_electrode_map.astype({'bank_number' : 'float', 
                                                        'TDT_channel' : 'float'})
        s_config = 'utah'
        if verbose:
            print('Stimulation electrode map is using the utah array')
    
    # load the correct recroding electrode map
    if recording_bank in [1, 2, 3, 7]:
        col_names = ['electrode_number', 'pin_number', 'TDT_channel']
        rec_electrode_map = pd.read_excel('FMA_electrode_map_arrays_1235.xlsx', usecols='A:C', names=col_names)
        rec_electrode_map = rec_electrode_map.astype({'TDT_channel' : 'float'})
        r_config = 'fma'
        if verbose:
            print('Recording Map is in FMA 1 2 3 5')
    elif recording_bank in [8, 9]:
        col_names = ['electrode_number', 'pin_number', 'TDT_channel']
        rec_electrode_map = pd.read_excel('FMA_electrode_map_arrays_67.xlsx', usecols='A:C', names=col_names)
        rec_electrode_map = rec_electrode_map.astype({'TDT_channel' : 'float'})
        r_config = 'fma'
        if verbose:
            print('Recording Map is in FMA 6 7')
    else:
        col_names = ['electrode_number', 'pin_number', 'recorded_channel', 'bank_number', 'TDT_channel']
        rec_electrode_map = pd.read_excel('Utah_electrode_map_array_4.xlsx', usecols='A:E', names=col_names)
        rec_electrode_map = rec_electrode_map.query('bank_number == {}'.format(recording_bank)).reset_index()
        rec_electrode_map = rec_electrode_map.astype({'bank_number' : 'float', 
                                                      'TDT_channel' : 'float'})
        r_config = 'utah'
        if verbose:
             print('Recording electrode map is using utah array')
    
    return stim_electrode_map, rec_electrode_map, stimulation_location, recording_location, s_config, r_config


def utah_stim_spatial(delta_heatmap, rec_channel):
    
     """ Function to plot the changes in threshold crossings based on the spatial arangement of the electodes
        (for Utah array) for stimulation channels
    
    Inputs 
    ------
    delta_heatmap : pandas pivot table
        Sorted pivot table that has the average change in threshold crossings per stimulation channel and recording electrode
        
    rec_channel : int
        The recording channel used to plot the spatial arangements of stimulations
        Suggested inputs:
            delta_heatmap.loc[rec_channel].name    - specific recording channel (will give error if it is not available)
            delta_heatmap.iloc[rec_channel].name   - recording channel by index of the heatmap
                                                            e.g. delta_heatmap.iloc[0] gives the top channel
                                                                 delta_heatmap.iloc[31] gives the bottom channel
            delta_heatmap.sample().index           - random recording channel
            
    s_config : string
        Basically a way to determine if the plot will be made on the Utah array or the FMA array
    
    Outputs
    -------
    spatial_array : numpy array
        spatial array with the number of threshold crossings for each corresponding electrode
    """
    
    
    # Get the data and change column name to avoid errors
    thr_delta = pd.DataFrame(delta_heatmap.loc[rec_channel]).reset_index()
    thr_delta = thr_delta.rename(columns={thr_delta.columns[1] : 'delta'})
    
    # This is the map of electrode locations based on the documentation for the utah array (this can change depending on each implanted array)
    # The idea is that each electrode number here will be replaced by the number of threshold crossings for that given electrode
    
    A = np.array([[np.nan, 88, 78, 68, 58, 48, 38    , 28    , 18, 32],
                  [96    , 87, 77, 67, 57, 47, 37    , 27    , 17, 8 ],
                  [95    , 86, 76, 66, 56, 46, 36    , 26    , 16, 7 ],
                  [94    , 85, 75, 65, 55, 45, 35    , 25    , 15, 6 ],
                  [93    , 84, 74, 64, 54, 44, 34    , 24    , 14, 5 ],
                  [92    , 83, 73, 63, 53, 43, 33    , 23    , 13, 4 ],
                  [91    , 82, 72, 62, 52, 42, np.nan, np.nan, 12, 3 ],
                  [90    , 81, 71, 61, 51, 41, 31    , 21    , 11, 2 ],
                  [89    , 80, 70, 60, 50, 40, 30    , 20    , 10, 1 ],
                  [np.nan, 79, 69, 59, 49, 39, 29    ,19     , 9 , 22]])

    # Create a dictionary to replace the values
    keys = thr_delta['stimulation_electrode']
    values = thr_delta['delta']
    d = dict(zip(keys, values))
    
    # The replacement
    spatial_array = np.copy(A)
    for electrode, metric in d.items():
        spatial_array[A == electrode] = metric
    
    # Replace the values not being used by nan
    all_channels = np.arange(97) # integers from 0-96. 
    unused = [i for i in all_channels if not i in thr_delta['stimulation_electrode'].unique()]
    
    for electrode in unused:
        spatial_array[A == electrode] = np.nan
    
    # np.repeat to get plot to be wider else delete that line and uncomment the next one
    # return spatial_array
    return np.repeat(np.repeat(spatial_array, 2, axis=1), 2, axis=0)


def utah_rec_spatial(delta_heatmap, stim_channel):
    
    """ Function to plot the changes in threshold crossings based on the spatial arangement of the electodes
        (for Utah array) for recording channels
    
    Inputs 
    ------
    delta_heatmap : pandas pivot table
        Sorted pivot table that has the average change in threshold crossings per stimulation channel and recording electrode
        
    stim_channel : int
        The stimulation channel used to plot the spatial arangements 
        
        Suggested inputs:
            delta_heatmap[stim_channel].name    - specific channel
            delta_heatmap.columns[0]               - most inhibitory channel
                                                     since they are sorted, 0 is most inhibitory, 1 is the next most inhibitory etc.
            delta_heatmap.columns[-1]              - most excitatory channel
                                                     since they are sorted, -1 is most excitatory, -2 is the next most excitatory etc. 
    
    Outputs
    -------
    spatial_array : numpy array
        spatial array with the number of threshold crossings for each corresponding electrode
    """

    # Get the data and change column name to avoid errors
    thr_delta = pd.DataFrame(delta_heatmap[stim_channel]).reset_index()
    thr_delta = thr_delta.rename(columns={thr_delta.columns[1] : 'delta'})
    
     # This is the map of electrode locations based on the documentation for the utah array (this can change depending on each implanted array)
    # The idea is that each electrode number here will be replaced by the number of threshold crossings for that given electrode
    
    A = np.array([[np.nan, 88, 78, 68, 58, 48, 38    , 28    , 18, 32],
                  [96    , 87, 77, 67, 57, 47, 37    , 27    , 17, 8 ],
                  [95    , 86, 76, 66, 56, 46, 36    , 26    , 16, 7 ],
                  [94    , 85, 75, 65, 55, 45, 35    , 25    , 15, 6 ],
                  [93    , 84, 74, 64, 54, 44, 34    , 24    , 14, 5 ],
                  [92    , 83, 73, 63, 53, 43, 33    , 23    , 13, 4 ],
                  [91    , 82, 72, 62, 52, 42, np.nan, np.nan, 12, 3 ],
                  [90    , 81, 71, 61, 51, 41, 31    , 21    , 11, 2 ],
                  [89    , 80, 70, 60, 50, 40, 30    , 20    , 10, 1 ],
                  [np.nan, 79, 69, 59, 49, 39, 29    ,19     , 9 , 22]])

    # Create a dictionary to replace the values
    keys = thr_delta['recording_electrode']
    values = thr_delta['delta']
    d = dict(zip(keys, values))
    
    # The replacement
    spatial_array = np.copy(A)
    for electrode, metric in d.items():
        spatial_array[A == electrode] = metric
    
    # Replace the values not being used by nan
    all_channels = np.arange(97) # integers from 0-96. 
    unused = [i for i in all_channels if not i in thr_delta['recording_electrode'].unique()]
    
    for electrode in unused:
        spatial_array[A == electrode] = np.nan
    
    # np.repeat to get plot to be wider else delete that line and uncomment the next one
    # return spatial_array
    return np.repeat(np.repeat(spatial_array, 2, axis=1), 2, axis=0)




def plot_spatials(delta_heatmap=None, channel=None, s_config=None, r_config=None, electrode_values=None):
    """ This function will return heatmap values for the different combinations. 
        In general, using electrode_values = delta_heatmap[column_no] is most used
    
    delta_heatmap : pandas pivot table
        Sorted pivot table that has the average change in threshold crossings per stimulation channel and recording electrode
    
    channel : int
        The channel used to plot the spatial arangements of stimulations 
        
    s_config/r_config : string
        Basically a way to determine if the plot will be made on the Utah array or the FMA array
        ** See fma_stim_spatial(), utah_stim_spatial(), fma_rec_spatial(), utah_rec_spatial()
        
    electrode_values : 32 x 1 array
        Electrode values to plot for an FMA array. 
        
    Output
    ------
    Formatted heatmap values to be used with sns.heatmap() for instance
    """
    
    # This will plot the data if it is not stored in a Pandas dataframe
    
    if isinstance(delta_heatmap, pd.DataFrame):
        if s_config =='fma':
            return fma_stim_spatial(delta_heatmap, channel)
        elif s_config == 'utah':
            return utah_stim_spatial(delta_heatmap, channel)
        elif r_config == 'fma':
            return fma_rec_spatial(delta_heatmap, channel)
        elif r_config == 'utah':
            return utah_rec_spatial(delta_heatmap, channel)
        else:
            print("There's a mistake somewhere, return is None")
            return None
    elif electrode_values is not None:
        return plot_fma(electrode_values)
    else:
        print("There's a mistake somewhere, return is None")

    
############### 
# Functions for plotting without a dataframe on an FMA
def plot_fma(fma_values):
    mapped_electrodes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 
                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    A = np.array([[28, np.nan, 10, np.nan],
     [28, np.nan, 10, np.nan],
     [29, np.nan, 11, np.nan],
     [29,     20, 11,      2],
     [30,     20, 12,      2],
     [30,     21, 12,      3],
     [31,     21, 13,      3],
     [31,     22, 13,      4],
     [32,     22, 14,      4],
     [32,     23, 14,      5],
     [33,     23, 15,      5],
     [33,     24, 15,      6],
     [34,     24, 16,      6],
     [34,     25, 16,      7],
     [35,     25, 17,      7],
     [35,     26, 17,      8],
     [np.nan, 26, np.nan,  8],
     [np.nan, 27, np.nan,  9],
     [np.nan, 27, np.nan,  9]])
    
    # Create a dictionary to replace the values
    keys = mapped_electrodes
    values = fma_values
    d = dict(zip(keys, values))
    
    # The replacement
    spatial_array = np.copy(A)
    for electrode, metric in d.items():
        spatial_array[A == electrode] = metric
    
    
    # np.repeat to get plot to be wider else delete that line and uncomment the next one
    # return spatial_array
    return np.repeat(spatial_array, 2, axis=1)


#-------------------------------------------------------------------------------------
# Mock online functions
#-------------------------------------------------------------------------------------

def get_online_heatmaps(df, column_name = 'delta_rate', ntrials=10, sort=False): 
    """ Get heatmaps that compare the change in the threshold crossing rates based on 
    the electrode number and the stimulation channel. 
        By default it will not be sorted to to give most inhibitory to most excitatory, 
        but the option is there. 
        
    This differs from "get_delta_heatmap()" as it will an array of 11 heatmaps. The first
    heatmap has the mean (same output as "get_delta_heatmap()") but the other 10 will consider
    heatmaps for the first/second/third/etc. time each stimulation channel is used. This should
    replicate an online experiment a bit better
      
      
    Input
    -----
    df : pandas dataframe
        Dataframe is from the output of MUA_stream_extract()
        It contains threshold crossings from a given block and stream
        In this case, the dataframe must contain the column 'stim_channel_order' or the code will break
        
    column_name : string
        Name of the column to use as the values to create the heatmap
        Defaults to 'rates_diff' as that is the one that I use the most
        
    ntrials : int
        Number of trials for each combination of stimulation/recording electrodes
        By default it is 10, as most experimental data used had 10
        
    sort : bool
        A boolean parameter to determine whether or not the heatmap should be sorted from most
        inhibitory to most excitatory
        False/0: do not sort
        True /1: sort
    """
    
    # Get the heatmap of the mean response
    heatmap_mean = get_delta_heatmap(df, column_name = column_name, sort=False)
    
    # Create a heatmap for each trial. The idea is to get the heatmap for each stimulation/recording electrode
    # combination for each trial. E.g. what was the first/second/third/ effect when that combination was stimulated.
    
    # initialize array
    heatmaps = [np.empty((32,32)) for x in range(ntrials)] 
    # populate array
    for i in range(ntrials):
        heatmaps[i] = get_delta_heatmap(df.loc[df.stim_channel_order == i+1], column_name=column_name, sort=False)
    
    # If sorting by most inhibitory to most excitatory
    if sort:
        # mean
        heatmap_mean = heatmap_mean.reindex(heatmap.T.mean().sort_values(ascending=False).index, axis=0)
        heatmap_mean = heatmap_mean.reindex(heatmap.mean().sort_values().index, axis=1)
        
        # Trial heatmaps
        for hm in heatmaps:
            hm = hm.reindex(heatmap.T.mean().sort_values(ascending=False).index, axis=0)
            hm = hm.reindex(heatmap.mean().sort_values().index, axis=1)
            
    return heatmap_mean, heatmaps
    

#----------------------------------------------------------------------------------------------------------------
# Plotting results from the GP
#----------------------------------------------------------------------------------------------------------------
def plot_individual_trials(df, column_number, column_name = 'rates_diff', trial_plot_title=None):
    """ Plot the results of individual stimulations and compare with the mean result of the stimulation
    
    Inputs
    ------
    df : pandas dataframe
        Contains data from the multiunit activtiy and can relate to electrode numbers
        Output from MUA_stream_extract() and append_electrodes()
    
    column_number : int
        Column of the heatmaps to plot. Corresponds to electrode numbers
        They are [2, 17], [20, 35]
    
    column_name : string
        Name of the column from the heatmaps (from the dataframe) to plot
        Recommended: - 'rates_delta'
                     - 'crossing_rate'
                     - 'BCR_local'
    trial_plot_title : string
        Title to put in the title of the plot showing the different trials
    
    Outputs : none
    """

    if not column_number in df.stimulation_electrode.unique():
        print('The chosen stimulation electrode is not correct. Please verify inputs')
        return
    
    # Get the heatmaps
    heatmap_mean, heatmaps = get_online_heatmaps(df, column_name)
    
    # Create looping variables
    a = np.arange(10).reshape(2, 5)
    trial_values = [np.empty((1, 32)) for x in range(10)]
    
    # Get the data to plot
    mean_values = heatmap_mean[column_number].values
    for i in range(10):
        trial_values[i] = heatmaps[i][column_number].values
        
    # Colour bar values
    cbar_min = np.nanmin((np.vstack((mean_values, trial_values))))
    cbar_max = np.nanmax((np.vstack((mean_values, trial_values))))
    
    # Plot the different trials
    fig, ax = plt.subplots(2, 5, figsize=(12,8))
    for i in range(2):
        for j in range(5):
            spat_map = plot_spatials(electrode_values = trial_values[a[i, j]])
            im = ax[i, j].imshow(spat_map, cmap='YlGnBu', vmin=cbar_min, vmax = cbar_max)
            ax[i, j].set(xticks=([]), yticks=([]), title = 'Trial {}'.format(a[i, j]+1))
            
    fig.colorbar(im, ax=ax.ravel().tolist(), location='right')
    fig.suptitle(trial_plot_title)
    
    # Plot the mean of trials
    fig, ax = plt.subplots(figsize=(6, 6))
    mean_map = plot_spatials(electrode_values = mean_values)
    im2 = ax.imshow(mean_map, cmap='YlGnBu', vmin=cbar_min, vmax=cbar_max)
    ax.set(xticks=([]), yticks=([]), title='Mean of All Trials')
    fig.colorbar(im2)
    
    
    
