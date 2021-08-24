import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from helpers import *
from collections import defaultdict
from scipy import signal, integrate
from sklearn.preprocessing import StandardScaler

################# Loading the data
def load_LFP_data(mat_file_path, mode):
    """ 
    Load a mat file and get the indices of stimulation as a dictionary
    
    Inputs
    ------
    mat_file_path : string
        path to mat file containing the LFP data
        
    mode : string
        Basically to load LFP1 or LFP2 data. This must be either 'LFP1' or 'LFP2', else the code will not work
        
    Outputs
    -------
    LFP : dictionary
        Contains the LFP data
        
    stim_ind : numpy array
        Contains indices of the stimulations
    """
    # Load the data
    LFP = loadmat(mat_file_path)
    # Remove headers
    LFP = LFP['consolidatedData']
    
    # Getting stimulation indices
    fs = LFP['samplingFreq']
    len_LFP = len(LFP[mode][0,:])
    time_LFP = np.linspace(1, len_LFP, len_LFP)/fs
    
    # Initialize array to store data
    stim_ind = np.zeros((len(LFP['stimTime'])), dtype=np.int)
    
    for i in range(len(LFP['stimTime'])):
        stim_ind[i] = np.argmax(LFP['stimTime'][i] <= time_LFP)
        
    return LFP, stim_ind



################# Signal processing functions

def z_score(x):
    """
    Function to z-score (zero-mean and unit variance) a signal given in x
    # Adapted from P.Sirpal (https://github.com/sirpalp)
    
    Input
    -----
    x: 1d vector
        Signal to z-score
        
    Output
    ------
    outputs the z-score signal
    """
    x = x.reshape(1, -1)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(x.T).T
    return Xz.flatten()


def extract_baseline(raw_signal, fs, stim_ind, baseline_time=100):
    """ Extract the baseline of the signal before each stimulation train.
    
    Inputs
    ------
    raw_signal : 1-d array
        Signal to get the stimulation responses from (should only be from one channel)
        This should be z-scored but can still work without it. 
        
    fs : float/double
        Sampling frequency of <raw_signal>
        
    stim_ind : 1-d int array
        Obtained by getting the indices from load_LFP_data()
        ** This MUST be an array (even if there is only 1 value)
        
    baseline_time : float/double
        The time (ms) to consider as the baseline signal before each pulse train. 
        
    Outputs
    -------
    
    baseline : [len(stim_ind) x bl_len] 2d array
        Contains sampled windows from the raw_signal that are concatenated together to give a baseline
        Useful for determining a consistant mean baseline signal to remove from the signal or to use for thresholds.
        
    inds : [len(stim_ind) x bl_len] 2d array
        Contains the indices from the baseline extraction
    """
    # Get the window length for the baseline signal (time --> indices)
    bl_len = np.int(np.ceil(baseline_time * fs) /1000)
    
    # Create an array to store the baseline response
    baseline = np.zeros((stim_ind.size, bl_len))
    inds = np.zeros_like(baseline)
    
    if stim_ind.size > 1:
        for i in range(len(stim_ind)):
            baseline[i, :] = raw_signal[stim_ind[i] - bl_len : stim_ind[i]]
            inds[i, :] = np.arange(stim_ind[i] - bl_len, stim_ind[i])
    else:
        baseline = raw_signal[stim_ind - bl_len : stim_ind]
        inds = np.arange(stim_ind - bl_len, stim_ind)
        
    return baseline, inds  


def extract_stim_response(raw_signal, fs, stim_ind, wtime, art_time=0, baseline=np.empty([])):
    """
    Extract the response from stimulation with the following considerations:
     - Time after the stimulation to consider as the artifact
     - Time before the stimulation to consider as the baseline (if removing the baseline from the signal)
     
     By deault the baseline is not removed, and not considering artifacts
    
     raw_signal: 1-d array
        Signal to get the stimulation responses from (Should only be from one channel)
        This should be zscored but can still work without it. 
        
    fs: float/double
        Sampling frequency of the signal in <stimulation data>
    
    stim_ind : 1-d int array
        Obtained by getting the indices from load_LFP_data()
    
    wtime: double
        The window length (in ms) to consider after the stimulation
        Should be less than: (duration between stimulations) - (baseline length)
        
    art_time: double
        If not zero, the time (ms) to consider the raw signal to be dominated by stimulation artifacts
        
    baseline: 1-d array
        If not empty, a sampled array containing the baseline obtained from extract_baseline()
        It is important to be consistent with the input of that function. If it is zscored, then <raw_signal>
        should be zscored as well. 
        
    Outputs
    -------
    stim_responses: [len(stim_ind) x wlen] 2d array
        Contains the signal following stimulations (varies based on the parameters given)
    """
    
    # Get the window lengths
    art_len = np.int(np.round(art_time * fs  / 1000))
    wlen = np.int(np.floor(wtime * fs / 1000))
    
    # Create array to store the response
    stim_responses = np.zeros((stim_ind.size, wlen+1))
    
    # Extract portions of the signal after stimulations
    if art_time == 0:
        
        # Removing basline
        if not baseline.size:
            for i in range(len(stim_ind)):
                stim_responses[i, :] = raw_signal[stim_ind[i] : stim_ind[i] + wlen + 1]
                stim_responses[i, :] = stim_responses[i, :] - np.mean(baseline[i, :])
       
        else: # Not removing the baseline
            for i in range(len(stim_ind)):
                stim_responses[i, :] = raw_signal[stim_ind[i] : stim_ind[i] + wlen + 1]
                
    # Ignore signal deemed to be dominated by the artifact
    else:
        # Removing the baseline
        if not baseline.size:
            for i in range(len(stim_ind)):
                stim_responses[i, art_len:] = raw_signal[stim_ind[i]+art_len : stim_ind[i]+wlen+1]
                stim_responses[i, art_len:] = stim_responses[i, art_len:] - mp.mean(baseline[i, :])
                stim_responses[i, :art_len] = 0
            
        else:
            for i in range(len(stim_ind)):
                stim_responses[i, art_len:] = raw_signal[stim_ind[i]+art_len : stim_ind[i]+wlen+1]
                stim_responses[i, :art_len] = 0
                
    return stim_responses




def get_bandpower(x, fs, band, nperseg=1024):
    """
    Inspired by Raphael Vallat's (UCB - Postdoc) code: https://raphaelvallat.com/bandpower.html
    output the bandpower for a given band (e.g. alpha, beta, gamma, etc.)
    
    Inputs
    ------
    x : 1-d array  
        The signal to extract band power from
    fs : double
        Sampling frequency
    band : 1 x 2 array
        Contains the frequencies to consider for obtaining the bandpower
        Examples:
        delta = [1, 4]         || theta = [4, 8]    || alpha = [8, 12]
        beta =  [12, 30]       || gamma = [30, 150] || low_gamma = [30, 70]
        high_gamma = [70, 150] || all = [1, 150]
    nperseg : int
        Length of each segment in scipy.welch, default is 256 
        
    Outputs
    -------
    bandpower: 
        bandpower in uV^2
    """
    # Get the band frequencies
    if (len(band) !=2):
        print('Please check the band input variable')
        return
        
    # Compute the periodogram
    freqs, psd = signal.welch(x, fs, nperseg=nperseg)
        
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    
    # Approximate area under periodogram for deisred band range 
    try:
        bandpower = integrate.simps(psd[idx_band], dx=freq_res)
    except: # There is not enough data points (usually a problem at low frequencies)
        bandpower = np.nan # or zero 
        
    return bandpower


###### Functions to set up plots
def append_electrodes(df, mode, channel_name = 'channel'):
    """
    Append electrodes to a dataframe based on the channel for visualization.
    This function requries maps between the electrode and the recording channels to be in the same folder as the notebook
    
    Inputs
    ------
    df : pandas dataframe
        The data frame to append the electrodes to
        
    mode : string
        Basically to load Utah Array or FMA configuations
        This must be 'utah' or 'fma' or function will not work
    
    channel_name: string
        Name of the column in df to merge on.
        Defaults to 'channel'
        
        
    Output
    ------
    df : pandas data frame
        Input dataframe with the electrodes appended according to the excel files referenced in the function
    """
    
    col_names = ['electrode', 'pin_number', 'recorded_channel']
    
    if mode == 'utah':
        electrode_map = pd.read_excel('utah_electrode_channel_map.xlsx', usecols='A:C', names=col_names)
    elif mode == 'fma':
        electrode_map = pd.read_excel('fma_electrode_channel_map.xlsx', usecols='A:C', names=col_names)
    else:
        print('The mode is incorrect. No appends performed. Please verify.')
        return df
    
    # Merge to get the recording electrodes
    df = pd.merge(left=df, right=electrode_map, how='left', left_on=channel_name, right_on='recorded_channel')
    
    # Drop redundant channels
    df = df.drop(columns=['pin_number', 'recorded_channel'])
    
    return df


################## Plotting Functions
def get_utah_spatial_map(df, column_name):
    """
    This function prepares a heatmap to plot when using data recorded in the Utah Array. 
    
    Inputs
    ------
    df : pandas dataframe
        The data frame to take the data from
    
    column_name : string
        The column to use to plot (e.g. band-power)
        
    Output
    ------
    Returns an array that can be fed into a plotting function based on the Utah Array used in recording.
    """
    
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
    keys = df['electrode']
    values = df[column_name]
    d = dict(zip(keys, values))
    
     # The replacement
    spatial_array = np.copy(A)
    for electrode, metric in d.items():
        spatial_array[A == electrode] = metric
    
    
    # Replace the values not being used by nan
    all_channels = np.arange(97) # integers from 0-96. 
    unused = [i for i in all_channels if not i in df['electrode'].unique()]
    
    for electrode in unused:
        spatial_array[A == electrode] = np.nan
    
    # np.repeat to get plot to be wider else delete that line and uncomment the next one
    # return spatial_array
    return np.repeat(np.repeat(spatial_array, 2, axis=1), 2, axis=0)


def get_fma_spatial_map(df, column_name):
    """
    This function prepares a heatmap to plot when using data recorded in the FMA. 
    
    Inputs
    ------
    df : pandas dataframe
        The data frame to take the data from
    
    column_name : string
        The column to use to plot (e.g. band-power)
        
    Output
    ------
    Returns an array that can be fed into a plotting function based on the Utah Array used in recording.
    """
    
    # This is the map of electrode locations based on the documentation for the utah array (this can change depending on each implanted array)
    # The idea is that each electrode number here will be replaced by the number of threshold crossings for that given electrode
    
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
    values = df[column_name]
    d = dict(zip(keys, values))
    
     # The replacement
    spatial_array = np.copy(A)
    for electrode, metric in d.items():
        spatial_array[A == electrode] = metric
    
    
    # Replace the values not being used by nan
    all_channels = np.arange(97) # integers from 0-96. 
    unused = [i for i in all_channels if not i in df['electrode'].unique()]
    
    for electrode in unused:
        spatial_array[A == electrode] = np.nan
    
    # np.repeat to get plot to be wider else delete that line and uncomment the next one
    # return spatial_array
    return np.repeat(spatial_array, 2, axis=1)