import tdt
import os 
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy import signal, integrate, stats
from scipy import interpolate
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from itertools import product
from multi_unit_plotting_functions import *



################# LOADING THE DATA
## Just to avoid copying code over and over again

def load_block_stim_data(block_number, block_path, root_dir = os.getcwd(), two_stims = False, verbose = True):
    """ This function loads the stimulation data from a TDT block and stores it in a dictionary. It will also take the correct
        data stored in text files (e.g. number of pulses, stimulation channels) and store them in the dictionary as well. 
    
        It requires this file: BackupStimoFiles/List_matching_blocks.xlsx to function. This excel file contains information
        about the blocks and which text files are associated with a given TDT block. 
        
        It equally requires defaultdict from the collections library to make a dictionary that is updated recursively
        
        Inputs
        ------
        block_number : int
            TDT block number to load the data from
        
        block_path : string
            Path to the TDT block. 
        
        root_dir : string
            Path to the root folder for the data. 
            It defaults to where the code is being run from. 
        
        two_stims : bool
            Bool parameter to use if there are stimulations from two channels at the same time. This will change some variable names 
            By default it is set to false. 
        
        verbose : bool
            Bool parameter to print what is happening. 
            Default is true
        
        Outputs
        -------
        data : TDT data structure
            This is the loaded TDT block data
        
        stimulation_data : collections.defaultdict()
            This is a dictionary that contains information about the stimulations from the txt files, etc. 
        
    
    """
    # Go to main folder
    os.chdir(root_dir)

    # This function opens the excel file for matching blocks
    # This is to know which txt files to open to get information
    
    # Opening the excel in to pandas dataframe and clean up the dataframe
    matching_path = 'BackupStimoFiles/Lists_matching_blocks.xlsx'
    columns = ['Block', 'Drop', 'List', 'List_no', 'Drop2', 'Info']
    matching = pd.read_excel(matching_path, names = columns)
    matching = matching.drop(columns=['Drop', 'Drop2'])
    matching.List = matching.List.str.strip() # Remove spaces at the end
    
    # Use the data frame to determine which files to open
    list_name = matching.loc[matching.Block == block_number].List.item()
    list_number = matching.loc[matching.Block == block_number].List_no.item()

    # Sanity check
    if verbose:
        print('The data for Block {} is in the {} folder with files ending in _{}'.format(block_number, list_name, list_number))
    
    
    # open the relevant files for the stimulation information
    os.chdir(('BackupStimoFiles/{}'.format(list_name))) # Goes to the correct folder

    txt_files = [f for f in os.listdir('.') if f.endswith(('_{}.txt').format(list_number))]

    for fname in txt_files:
        with open(fname, 'r') as f:
            if fname.startswith('Current1'):
                current1 = np.loadtxt(f.readlines())
                current1 = current1[:-1]
            if fname.startswith('Current2'):
                current2 = np.loadtxt(f.readlines())
                current2 = current2[:-1]
            if fname.startswith('ISDL'):
                ISDL = np.loadtxt(f.readlines())
                ISDL = ISDL[:-1]
            if fname.startswith('Nb1'):
                Nb1 = np.loadtxt(f.readlines())
                Nb1 = Nb1[:-1]
            if fname.startswith('Nb2'):
                Nb2 = np.loadtxt(f.readlines())
                Nb2 = Nb2[:-1]
            if fname.startswith('Stim1'):
                stim1 = np.loadtxt(f.readlines())
                stim1 = stim1[:-1]
            if fname.startswith('Stim2'):
                stim2 = np.loadtxt(f.readlines())
                stim2 = stim2[:-1]
                
    
    # Create a new list that lists the order of occurence for the stimulation channel (to be used for mock online experiments)
    unique_channels = np.unique(stim2)
    stim_channel_order = np.zeros_like(stim2)
    
    for channel in unique_channels:
        indices = np.where(stim2 == channel)[0]
        
        for i in range(len(indices)):
            stim_channel_order[indices[i]] = i+1
    
    
    # Go to main folder
    os.chdir(root_dir)
    
    # Read the block
    if verbose:
        print('Loading TDT block data')
    data = tdt.read_block(block_path)
    
    ## Storing the stimulation data into a dictionary
    # Careful, it is downsampled to [::4]
    
    # Get the stimulation indices (downsmapled)
    if not two_stims:
        # Get all indices from the stimulation channel that are nonzero
        stim_indices = np.argwhere(data.streams.STM_.data[3, :][::4] != 0).flatten()
    else:
        # Get all indices from the two stimulation channels that are nonzero
        stim_indices1 = np.argwhere(data.streams.STM_.data[2, :][::4] !=0).flatten()
        stim_indices2 = np.argwhere(data.streams.STM_.data[3, :][::4] !=0).flatten()
    
    # Get the indices from channel that preps stimulations
    # (It is around 50 ms before each stimulation train)
    stim_starts = np.argwhere(np.diff(data.streams.STM_.data[0, :][::4])>0).flatten()
    # Put +1 back because of np.diff
    stim_starts = stim_starts +1
    # Might be unique to block 3 but need to remove the first 2 as they are not protocol related
    stim_starts = stim_starts[2:] 
    # Add another point for how I envision the loop to go
    stim_starts = np.append(stim_starts, len(data.streams.STM_.data[0,:][::4]))

    # To make nested dictionaries
    tree = lambda: defaultdict(tree)

    stimulation_data = tree()

    for i in range(len(stim_starts)-1):
        # Store the stimulation channel, amplitude, interchannel delay, npulses
        # These are stored in the txt files loaded earlier
        stimulation_data[i]['stim_amp1'] = current1[i]
        stimulation_data[i]['stim_amp2'] = current2[i]
        stimulation_data[i]['ISDL'] = ISDL[i]
        stimulation_data[i]['num_pulses1'] = Nb1[i]
        stimulation_data[i]['num_pulses2'] = Nb2[i]
        stimulation_data[i]['stim_channel1'] = stim1[i]
        stimulation_data[i]['stim_channel2'] = stim2[i]

        # Order of stimulation channel
        stimulation_data[i]['stim_channel2_order'] = stim_channel_order[i]
        
        # Sampling frequency
        stimulation_data[i]['fs'] = data.streams.Wave.fs

        # Store the indices of stimulation for a given pulse train
        if not two_stims:
            stimulation_data[i]['pulse_indices'] = stim_indices[(stim_indices >= stim_starts[i]) & (stim_indices <= stim_starts[i+1])]
        else:
            temp1 = stim_indices1[(stim_indices1 >= stim_starts[i]) & (stim_indices1 <= stim_starts[i+1])] 
            temp2 = stim_indices2[(stim_indices2 >= stim_starts[i]) & (stim_indices2 <= stim_starts[i+1])]
            stim_indices = np.sort(np.concatenate((temp1, temp2)))
            stimulation_data[i]['pulse_indices'] = stim_indices
        
        stimulation_data[i]['stim_start_index'] = stim_starts[i]

        # Store the time associated with the stimulation start, first & last pulses
        stimulation_data[i]['pulse_start_time'] = stimulation_data[i]['pulse_indices'][0] / stimulation_data[i]['fs']
        stimulation_data[i]['pulse_end_time'] = stimulation_data[i]['pulse_indices'][-1] / stimulation_data[i]['fs']
        stimulation_data[i]['stim_start_time'] = stimulation_data[i]['stim_start_index'] / stimulation_data[i]['fs']
        
        
    if verbose:    
        print('Stimulation data dictionary finished')    
        
    return data, stimulation_data



############################ Multi-unit functions

# Adapted from Parikshat
def z_score(x):
    x = x.reshape(1, -1)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(x.T).T
    return Xz.flatten()

def extract_baseline(raw_signal, fs, stimulation_data, baseline_time=50):
    """ Extract the baseline of the signal before each stimulation train.
    
    Inputs
    ------
    raw_signal : 1-d array
        Signal to get the stimulation responses from (should only be from one channel)
        This should be z-scored but can still work without it. 
        
    fs : float/double
        Sampling frequency of <raw_signal>
        
    stimulation_data : nested dictionary
        This contains all the stimulation information (channel, delay, pulse times, etc.)
        This needs to be at the same frequency as the raw signal (i.e. it is down-sampled)
        
    baseline_time : float/double
        The time (ms) to consider as the baseline signal before each pulse train. 
        
    Outputs
    -------
    
    baseline : 1-d (flattened) array
        Contains sampled windows from the raw_signal that are concatenated together to give a baseline
        Useful for determining a consistant mean baseline signal to remove from the signal or to use for thresholds.
        
    bl_total_time : float
        Contains the total amount of time sampled for the baseline. 
        Can be used to generate spike rates or threshold crossing rates
        
    inds : 1-d (flattened) array
        Contains the indices from the baseline extraction
    """
    
    # Get the window length for the baseline signal (time --> indices)
    bl_len = np.int(np.ceil(baseline_time * fs) /1000)
    
    # Get the indices for the first stimulation in each pulse train 
    pulse_starts = np.zeros(len(stimulation_data), dtype=np.int)
    
    for i in range(len(stimulation_data)):
        pulse_starts[i] = stimulation_data[i]['pulse_indices'][0]
        
    # Create an array to store the baseline response    
    baseline = np.zeros((len(stimulation_data), bl_len))    
    inds = np.zeros((len(stimulation_data), bl_len))
    
    for i in range(len(stimulation_data)):
        baseline[i, :] = raw_signal[pulse_starts[i]-bl_len : pulse_starts[i]]
        inds[i, :] = np.arange(pulse_starts[i]-bl_len, pulse_starts[i])
    
    return baseline.flatten(), inds.flatten()



def extract_signal(raw_signal, fs, stimulation_data, wtime, art_time=0, baseline=np.empty([])):
    """ Extract the desired signal after stimulation with the following considerations:
        - time after the stimulation to consider as the artifact
        - time before the stimulation to consider as the baseline signal
    
    Inputs
    ------
    raw_signal: 1-d array
        Signal to get the stimulation responses from (Should only be from one channel)
        This should be zscored but can still work without it. 
        
    fs: float/double
        Sampling frequency of the signal in <stimulation data>
    
    stimulation_data: nested dictionary
        This contains all the stimulation information (channel, delay, ,pulse times, etc.)
        As the sampling frequency is different than with the raw signal the entire dictionary is required
    
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
    stim_windows: [len(stimulation_data) x wlen] 2d array
        Contains the signal following stimulations (varies based on the parameters given)
    """
    
    # Get the window lengths
    art_len = np.int(np.round(art_time * fs /1000))
    wlen  = np.int(np.floor(wtime*fs/1000))
    
    # Get the indices for the first stimulation in each pulse train and last
    pulse_starts = np.zeros(len(stimulation_data), dtype=np.int)
    pulse_stops = np.zeros(len(stimulation_data), dtype=np.int)
    
    # To account for the delay between stimulation and the recorded data (as a result of downsampling)
    delay = 0 # by inspection [i.e. ignore the first 2.5 ms]
    
    for i in range(len(stimulation_data)):
        pulse_starts[i] = stimulation_data[i]['pulse_indices'][0]
        pulse_stops[i] = stimulation_data[i]['pulse_indices'][-1] + delay 
        
      
    # Create an array to store the response
    stim_responses = np.zeros((len(stimulation_data), wlen+1))
    
    # Extract the entire signal after the stimulation
    # Commented out print statements are useful for debugging
    if art_time ==0:
#         print('No artifact consideration')
        # Removing the baseline
        if not baseline.size:
#             print('Considering the baseline with no artifacts')
            for i in range(len(pulse_starts)):
                stim_responses[i, :] = raw_signal[pulse_stops[i] : pulse_stops[i]+wlen +1]
#                 baseline, _ = extract_baseline(raw_signal, fs, stimulation_data, baseline_time=baseline_time)
                stim_responses[i, :] = stim_responses[i, :] - np.mean(baseline)
        # Not removing the baseline
        else: 
#             print('No baseline consideration and no artifact consideration')
            for i in range(len(pulse_starts)):
                stim_responses[i, :] = raw_signal[pulse_stops[i] : pulse_stops[i]+wlen+1]
            
    else: 
#         print('Considering artifacts')
        # Removing the baseline
        if not baseline.size:
#             print('Considering baseline and artifacts')
            for i in range (len(pulse_starts)):
                stim_responses[i, art_len:] = raw_signal[pulse_stops[i]+art_len : pulse_stops[i] + wlen+ 1]
#                 baseline, _ = extract_baseline(raw_signal, fs, stimulation_data, baseline_time=baseline_time)
                stim_responses[i, art_len:] = stim_responses[i, art_len:] - np.mean(baseline)
                stim_responses[i, :art_len] = 0
        else:
#             print(' No baseline consideration, but considering artifacts')
            for i in range(len(pulse_starts)):
                stim_responses[i, art_len:] = raw_signal[pulse_stops[i]+art_len : pulse_stops[i] +wlen+1]
                stim_responses[i, :art_len] = 0
                
    return stim_responses


def MultUnit_bandpower(x, fs, nperseg=1024, min_freq=500):
    """
    Inspired by Raphael Vallat's (UCB - Postdoc) code: https://raphaelvallat.com/bandpower.html
    output the bandpower for a given band (e.g. alpha, beta, gamma, etc.)
    
    Inputs
    ------
    x : 1-d array  
        The signal to extract band power from
    fs : double
        Sampling frequency
    nperseg : int
        Length of each segment in scipy.welch, default is 256 
    min_freq : double/float
        Minimum frequency to consider for calculating the band power. 
        This should be the same frequency as the one used for the high-pass filter. 
        
    Outputs
    -------
    bandpower: 
        bandpower in uV^2
    """
    # Get the band frequencies
        
    # Compute the periodogram
    freqs, psd = signal.welch(x, fs, nperseg=nperseg)
        
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    idx_band = np.where(freqs > 500)[0]
    
    # Approximate area under periodogram for deisred band range 
    try:
        bandpower = integrate.simps(psd[idx_band], dx=freq_res)
    except: # There is not enough data points (usually a problem at low frequencies)
        bandpower = np.nan # or zero 
        
    return bandpower
    

def MultUnit_thr_cross_rate(x, hp_signal, fs, stimulation_data, baseline, wtime, nstdevs=3):
    """ Get the threshold crossing rate before and after stimulation defined by a N * std. dev. of the baseline before
    a stimulation event. The baseline / input should be passed through a high-pass filter (at 500 Hz).
    
    Inputs
    ------
    x : [(number of pulse trains) x (window size)] 2d array
        x contains the signal to analyze, typically obtained from extract_signal()
            NB: the amount of time after each pulse train to consider as signal/artifacts is done here
        x should only be the signal after stimulation trains
        x is what is being analyzed
     
    hp_signal : Nx1 double array, N=fs*recording time
        The high-passed signal
        It is used to determine the number of baseline crossings for comparison
        
    fs : double
        Sampling frequency of the data
        
    stimulation_data : nest dictionary
        Contains all the information for the stimulations (channel, delay between channels, pulse times, etc. )
        
    baseline : [(number of pulse trains) * (baseline window length)] 1d array
        baseline contains the signal prior to each stimulation train. It is typically obtained from extract_baseline()
        baseline is used to generate the threshold
    
    wtime : double
        The window length (in ms) to consider after the stimulation
        Should be less than: (duration between stimulations) - (baseline length)
        Make sure that this is the same as the length passed to x
    
    nstdevs : double
        The number of standard deviations from the mean baseline signal to create the threshold from
    
        
    Outputs
    -------
    crossings : N x 1 array, N = len(stimulation_data) = number of pulse trains
        Contains the number of threshold crossings after a pulse train
    """   
    
    # Create variables to store results
    crossings = np.zeros(len(stimulation_data))
    bl_crossings = np.zeros_like(crossings)
    crossing_rates = np.zeros(len(stimulation_data))
    
    # Threshold as defined 
    threshold = nstdevs * np.std(baseline) + np.mean(baseline)
    
    # Find the number of crossings after the stimulation
    for i in range(len(crossings)):
        # Count the number of times that the signal passes the threshold after stimulation (both positive and negative)
        events = np.where(np.abs(x[i, :]) >=threshold)[0]
        
        if len(events) > 0:
            idx = np.where(np.diff(events, prepend=0) > 1)[0] # Remove successive time points
            crossings[i] = len(events[idx])
        else:
            crossings[i]= 0
    
    
    crossing_rates = (crossings/wtime)*1000
    
    # For the baseline
    
    # Get the baseline threshold crossing rate
    baseline_time = (len(baseline)/len(stimulation_data))*1000/fs
    bl_events = len(np.where(np.abs(baseline) >= threshold)[0])

    bl_crossing_rate = bl_events / (baseline_time * len(stimulation_data)) * 1000 # pulses/sec
    
        
    return crossing_rates, bl_crossing_rate
    

    
def MultUnit_thr_crossings(x, hp_signal, fs, stimulation_data, baseline, nstdevs=3):
    """ Get the number of points that cross a threshold defined by 3 * std dev of the baseline before
    a stimulation event. The baseline / input should be passed through a high-pass filter (at 500 Hz).
    
    Inputs
    ------
    x : [(number of pulse trains) x (window size)] 2d array
        x contains the signal to analyze, typically obtained from extract_signal()
            NB: the amount of time after each pulse train to consider as signal/artifacts is done here
        x should only be the signal after stimulation trains
        x is what is being analyzed
     
    hp_signal : Nx1 double array, N=fs*recording time
        The high-passed signal
        It is used to determine the number of baseline crossings for comparison
        
    fs : double
        Sampling frequency of the data
        
    stimulation_data : nest dictionary
        Contains all the information for the stimulations (channel, delay between channels, pulse times, etc. )
        
    baseline : [(number of pulse trains) * (baseline window length)] 1d array
        baseline contains the signal prior to each stimulation train. It is typically obtained from extract_baseline()
        baseline is used to generate the threshold
    
    nstdevs : double
        The number of standard deviations from the mean baseline signal to create the threshold from
        
        
    Outputs
    -------
    crossings : N x 1 array, N = len(stimulation_data) = number of pulse trains
        Contains the number of threshold crossings after a pulse train
        
    bl_crossings : N x 1 array, N = len(stimulation_data) = number of pulse trains
        Contains the number of threshold crossings before a pulse train
        
    bl_mean : N x 1 array, N = len(stimulation_data) = number of pulse trains 
        Contains the mean value of the baseline.
        This is not very useful, but can help with trouble shooting. 
    """
    # Create variables to store results
    crossings = np.zeros(len(stimulation_data))
    bl_crossings = np.zeros_like(crossings)
    baseline_mean = np.zeros_like(crossings)
    
    # Threshold as defined 
    threshold = nstdevs * np.std(baseline) + np.mean(baseline)
    
    bl_window = int(len(baseline)/len(stimulation_data))
    
    # Find the number of crossings
    for i in range(len(crossings)):
        # Get start index for the pulse train (for baseline crossings)
        pulse_start = stimulation_data[i]['pulse_indices'][0]
        
        # Basline before each pulse train
        bl = hp_signal[pulse_start - bl_window : pulse_start]
        baseline_mean[i] = np.mean(bl)
        
        # Count the number of times that the signal passes the threshold after stimulation (both positive and negative)
        events = np.where(np.abs(x[i, :]) >=threshold)[0]
        
        if len(events) > 0:
            idx = np.where(np.diff(events, prepend=0) > 1)[0] # Remove successive time points
            crossings[i] = len(events[idx])
        else:
            crossings[i]= 0
        
        # Repeat with the baseline crossings
        bl_events = np.where(np.abs(bl) >= threshold)[0]

        if len(bl_events) > 0:
            idx = np.where(np.diff(bl_events, prepend=0) > 1)[0]
            bl_crossings[i] = len(bl_events[idx])
        else:
            bl_crossings[i] = 0
        
    return crossings, bl_crossings, baseline_mean


def MUA_stream_extract(stream, stimulation_data, wtime=200, baseline_time=300, art_time=2.9, nstdevs=2.75, stop_time=None, verbose=True, two_stims=False, keep_traces=False):
    """ This is a function to extract multiunit activity metrics from a given data stream. 
        The results are stored in a nested dictionary.
        
    Inputs
    ------
    stream : TDT data stream
        This is the TDT stream similar to the following examples:
        data.streams.Wave ;  data.streams.Wav2 ; data.streams.Wav3 ; data.streams.Wav4 ; 
        
    stimulation_data : nested dictionary
        This contains all the stimulation information (channel, delay, pulse times, etc.)
        This needs to be at the same frequency as the raw signal (i.e. it is down-sampled)
    
    wtime : float
        Time after the final stimulation pulses (incl. artifact time) to consider as the "signal" to examine
        It should be given in ms. 
    
    baseline_time : float
        Time before the stimulation trains to consider as the baseline signal
        It should be given in ms. 
    
    art_time : float
        This is the time to consider the signal to be dominated by the stimulation artifact. 
        It should be given in ms. 
        The default value is 2.9 ms with multiunit activity (by examination). This means that if wtime is 200 ms, 
        the signal considered is from 2.9-200 ms after the final pulse in the stimulation. 
        
    nstdevs : float
        This is the number of standard deviations to use to determine the threshold crossings. 
        The default value of 2.75 showed a good tradeoff between the effect seen after stimulation and getting a 
        reliable baseline. 
    
    stop_time : int
        This is the index to stop the data collection at in the event that there are some abnormalities in the signal. 
        
        
    verbose : bool
        Bool parameter to print what is happening. 
        Default is true
        
    two_stims : bool
        Bool parameter to use if there are stimulations from two channels at the same time. This will change some variable names 
        By default it is set to false. 
        
    keep_traces : bool
        Bool parameter to use if keeping the traces is desired
        They will be stored in a 3d array of size [n_channels x len(stimulation_data) x window_length]
        window length is wtime in indices. 
        
    Outputs
    -------
    results : PANDAS dataframe
        This contains the multi-unit activity results for a given channel. It has a variety of different metrics, 
        as well as different information for each individual pulse. 
        
    traces : 3d array of size [n_channels x len(stimulation_data) x window_length]
        Contains the traces of the high-passed signal
    """
    
    
    # Create a nested dictionary to store results from multi unit activity
    tree = lambda: defaultdict(tree)
    results = tree()
    
    # Looping variables
    fs = stream.fs
    b, a = signal.butter(2, 500, btype='hp', fs=fs)
    channels = np.arange(32)
    
    if verbose:
        print(str(stream.name))
        
        
    if keep_traces:
        wlen = np.int(np.floor(wtime * fs / 1000))
        traces = np.empty((len(channels), len(stimulation_data), wlen+1))
        
    for i, (channel) in enumerate(channels):
        
        #z_score and high-pass filter
        if stop_time is None:
            zwave = z_score(stream.data[channel, :])
        else:
            zwave = z_score(stream.data[channel, :stop_time])
            
        zwave_hp = signal.lfilter(b, a, zwave)

        # get baseline, signal after pulses
        baseline, _ = extract_baseline(zwave_hp, fs, stimulation_data, baseline_time=baseline_time)
        sig = extract_signal(zwave_hp, fs, stimulation_data, wtime=wtime, art_time=art_time, baseline=baseline)
        
        # Get threshold crossings and rates
        crossing_rates, bl_crossing_rate = MultUnit_thr_cross_rate(sig, zwave_hp, fs, stimulation_data, baseline, wtime=wtime, nstdevs=nstdevs)
        cr, bl, bl_mean = MultUnit_thr_crossings(sig, zwave_hp, fs, stimulation_data, baseline)

        for pulse in range(len(stimulation_data)):
            # Indices
            ind = i * len(stimulation_data) + pulse
            
            # If only using 1 stimulation channel in the block 
            if not two_stims:
                # Pulse information
                results[ind]['recording_channel'] = channel + 1
                results[ind]['stimulation_channel'] = np.int(stimulation_data[pulse]['stim_channel2'])
                results[ind]['pulse_no'] = pulse
                results[ind]['wtime'] = wtime
                results[ind]['window_length'] = np.shape(sig)[1]
                results[ind]['num_pulses'] = np.int(stimulation_data[pulse]['num_pulses2'])
                results[ind]['stim_amplitude'] = np.int(stimulation_data[pulse]['stim_amp2'])
                results[ind]['stim_channel_order'] = stimulation_data[pulse]['stim_channel2_order']
            
            else:
                # Pulse information
                results[ind]['recording_channel'] = channel + 1
                results[ind]['pulse_no'] = pulse
                results[ind]['wtime'] = wtime
                results[ind]['window_length'] = np.shape(sig)[1]
                
                # Stim channel 1 information
                results[ind]['stimulation_channel1'] = np.int(stimulation_data[pulse]['stim_channel1'])
                results[ind]['num_pulses1'] = np.int(stimulation_data[pulse]['num_pulses1'])
                results[ind]['stim_amplitude1'] = np.int(stimulation_data[pulse]['stim_amp1'])
                results[ind]['ISDL'] = np.float(stimulation_data[pulse]['ISDL'])
                
                # Stim channel 2 information
                results[ind]['stimulation_channel2'] = np.int(stimulation_data[pulse]['stim_channel2'])
                results[ind]['num_pulses2'] = np.int(stimulation_data[pulse]['num_pulses2'])
                results[ind]['stim_amplitude2'] = np.int(stimulation_data[pulse]['stim_amp2'])
                results[ind]['stim_channel_order2'] = stimulation_data[pulse]['stim_channel2_order']
                
                
            # Raw metrics
            results[ind]['n_crossings'] = cr[pulse]
            results[ind]['bl_crossings'] = bl[pulse]
            results[ind]['bl_mean'] = bl_mean[pulse] # 
            results[ind]['crossing_rate'] = crossing_rates[pulse]
            
            # Local baseline crossing rate (before the pulse train)
            results[ind]['BCR_local'] = bl[pulse]/baseline_time*1000
            
            # Average baseline crossing rate in a given channel
            results[ind]['BCR_global'] = bl_crossing_rate
            
            # Percentage difference compared to global baseline
            results[ind]['percentDifference_gBCR'] = 100*(results[ind]['crossing_rate'] - results[ind]['BCR_global'])/results[ind]['BCR_global']
            
            # Difference in rates (gBCR compares to global baseline, 2nd line to local baseline)
            results[ind]['delta_rate_gBCR'] = crossing_rates[pulse] - bl_crossing_rate
            results[ind]['delta_rate'] = results[ind]['n_crossings']/wtime*1000 - results[ind]['bl_crossings']/baseline_time*1000
            
              # Get the mean signal value after stimulation (not useful)
#             results[ind]['signal_mean'] = np.mean(sig[pulse][sig[pulse]!=0])
              # Get the mean signal value before stimulation (not useful)
#             results[ind]['bl_mean'] = bl_mean[pulse]
        if keep_traces:
            traces[i, :] = sig
        
    if verbose:        
        print('Storing in dataframe')

    df = pd.DataFrame.from_records(results).T
    
    if not two_stims:
        df = df.astype({'recording_channel' : 'float', 
                        'stimulation_channel' : 'float', 
                        'pulse_no' : 'int', 'wtime' : 'float',
                        'window_length' : 'float', 
                        'n_crossings' : 'float', 
                        'bl_crossings' : 'float', 
                        'crossing_rate'  : 'float', 
                        'BCR_local' : 'float', 
                        'BCR_global' : 'float', 
                        'percentDifference_gBCR' : 'float',
                        'delta_rate_gBCR' : 'float', 
                        'delta_rate' : 'float'})
    else:
        df = df.astype({'recording_channel' : 'float', 
                        'stimulation_channel1' : 'float',
                        'stimulation_channel2' : 'float', 
                        'pulse_no' : 'int', 'wtime' : 'float',
                        'window_length' : 'float', 
                        'n_crossings' : 'float', 
                        'bl_crossings' : 'float', 
                        'crossing_rate'  : 'float', 
                        'BCR_local' : 'float', 
                        'BCR_global' : 'float', 
                        'percentDifference_gBCR' : 'float',
                        'delta_rate_gBCR' : 'float', 
                        'delta_rate' : 'float'})
    
    if verbose:
        print('Finished')
        
    if keep_traces:
        return df, traces
    else:
        return df



def append_electrodes(df, stream, block_number, verbose = True, two_stims = False):
    """ This function appends the electrodes to the corresponding recording and stimulation channel
    based on the stream and block_number to the dataframe given as an input. 
    
    It uses the block number and stream to determine the correct mapping to use
    
    Inputs
    ------
    df : pandas dataframe
        This should be the output from MUA_stream_extract()
        I don't think the function will work otherwise. 
    
    # The next two need to be the same stream/block combination used to generate the dataframe from MUA_stream_extract()
    stream : TDT data stream
            e.g. data.streams.Wave ; data.streams.Wav2

    block_number : int
            The block number of the TDT data you are using
            
    verbose : bool
        Bool parameter to print what is happening. 
        Default is true
        
    two_stims : bool
        Bool parameter to use if there are stimulations from two channels at the same time. This will change some variable names 
        By default it is set to false. 
            
    Output
    ------
    df : dataframe
        This dataframe contains the information that was given as an input to the function as well as the corresponding stimulation electrode
        and recording electrode based on the stimulation and recording channels for a given pulse/trial. 
    """
    # This loads the correct map of the electrodes depending on the orientation of the Omnetics wire and the bank number
    stim_electrode_map, rec_electrode_map, stimulation_location, recording_location, s_config, r_config = get_electrode_maps(stream, block_number, verbose=verbose)

    # Merge to get recording electrodes
    df = pd.merge(left = df, right = rec_electrode_map, how = 'left',
                               left_on = 'recording_channel', right_on = 'TDT_channel')

    # Rename columns and drop some redundant columns
    df = df.rename(columns={'electrode_number' : 'recording_electrode'})
    df = df.drop(columns=['pin_number', 'TDT_channel'])

    # One stimulation channel
    if not two_stims:
        # Merge to get the stimulation electrodes
        df = pd.merge(left = df, right = stim_electrode_map, how='left',
                                   left_on = 'stimulation_channel', right_on = 'TDT_channel')

        # Rename columns and drop some redundant columns
        df= df.rename(columns={'electrode_number' : 'stimulation_electrode'})
        df = df.drop(columns=['pin_number', 'TDT_channel'])
    
    # Two stimulation channels
    else: 
        
        # Merge to get the stimulation electrode 2
        df = pd.merge(left = df, right = stim_electrode_map, how='left',
                                   left_on = 'stimulation_channel2', right_on = 'TDT_channel')
        
        # Rename columns and drop some redundant columns
        df= df.rename(columns={'electrode_number' : 'stimulation_electrode2'})
        df = df.drop(columns=['pin_number', 'TDT_channel'])
        
        
        # Merge to get the stimulation electrode 1
        df = df = pd.merge(left = df, right = stim_electrode_map, how='left',
                                   left_on = 'stimulation_channel1', right_on = 'TDT_channel')
        
        # Rename columns and drop some redundant columns
        df= df.rename(columns={'electrode_number' : 'stimulation_electrode1'})
        df = df.drop(columns=['pin_number', 'TDT_channel'])
        
      
        
    return df


