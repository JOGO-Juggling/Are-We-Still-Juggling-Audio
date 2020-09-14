import torch
import argparse
import librosa
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from retrieve_utils import AudioReader, VideoReader
from models import Convolutional


def get_energy_peaks(v_file, a_file, f_size=25, f_stride=10):
    '''Calculates the energy peaks for an audio file, and returns the timings
       (in ms) of the energy peaks detected. Returns a list of EP timings.'''
    audioreader = AudioReader(a_file, f_size, f_stride)
    videoreader = VideoReader(v_file)

    total_frames, framerate = videoreader.total_frames, videoreader.fps

    no_peak_amp, ste, epd = [0], [], []
    prev_ste0, prev_ste1 = 0, 0

    stop = int(total_frames / framerate) - 1

    for i, samples in enumerate(audioreader):
        if type(samples[0]) is np.ndarray:
            samples = samples.sum(axis=1) / 2 # Convert to 1 channel
        cur_ste = np.mean(np.abs(samples))
        ste.append(cur_ste)

        # Filter out all energy peaks < mean_amplitude of non-peaks
        mean_amplitude = np.mean(no_peak_amp)
        mean_condition = prev_ste1 > mean_amplitude

        if prev_ste1 > prev_ste0 and prev_ste1 > cur_ste and mean_condition:
            epd.append((i - 1) * f_stride)
        else:
            no_peak_amp.append(cur_ste) # Update non-peak amplitudes

        prev_ste0 = prev_ste1
        prev_ste1 = cur_ste

        # Stop 1 second before end of video
        if i * f_stride > stop * 1000:
            break
    
    return ste, epd


def get_mfccs(a_file, ep_timings, n_range=[-2, 3]):
    '''Constructs the MFCC feature matrix for each energy peak in an audio file
       given the timings of the energy peaks. Returns a list of MFCC matrices'''
    duration = librosa.get_duration(filename=a_file)
    signal, samplerate = librosa.load(a_file)
    mfccs = librosa.feature.mfcc(signal, samplerate).T
    ms_per_slice = (duration / len(mfccs)) * 1000
    epd_mfccs = []

    # Iterate trough energy peaks, match 
    for ep in ep_timings:
        mfcc_index = int((ep / ms_per_slice))

        if mfcc_index < len(mfccs) - 1:
            i,j = mfcc_index + n_range[0], mfcc_index + n_range[1]
            matrix = mfccs[i:j,2:14]
            epd_mfccs.append([np.abs(matrix)])
    return np.array(epd_mfccs), mfccs


def display_plots(a_file, ste, false_epd, true_epd, mfccs):
    '''Displays the results of the extraction functions in a plot.'''
    signal, samplerate = librosa.load(a_file)
    if type(signal[0]) is np.ndarray:
        signal = signal.sum(axis=1) / 2 # Convert to 1 channel

    fig, axs = plt.subplots(3, 1)
    ste_time = np.arange(len(ste)) * 10

    # Plot audio signal
    axs[0].plot(signal[:len(signal) // 10])

    # Plot STE and detected energy peaks
    ste_time, ste = ste_time[:len(ste_time) // 10], ste[:len(ste) // 10]
    mfccs = mfccs[:len(mfccs) // 10]

    axs[1].plot(ste_time, ste)
    axs[1].vlines(false_epd, [0] * len(false_epd), [max(ste)] * len(false_epd),
               color='r', linestyles='--')
    axs[1].vlines(true_epd, [0] * len(true_epd), [max(ste)] * len(true_epd),
               color='r', linestyles='--')
    axs[2].imshow(np.rot90(mfccs[:,2:14]))
    plt.show()


def main(video_file, audio_file, model, device):
    # Extract energy peaks
    short_term_energy, energy_peaks = get_energy_peaks(video_file, audio_file)

    # Extract MFCCs
    epd_mfccs, mfccs = get_mfccs(audio_file, energy_peaks)
    mfccs_batch = torch.FloatTensor(epd_mfccs)

    # Prepare model
    model.eval()
    model.to(device)

    # For each energy peak make prediction using MFCCs
    with torch.no_grad():
        y_hat = model(mfccs_batch.to(device)).cpu()
    _, predictions = torch.max(y_hat.data, 1)

    # Display predictions
    display_plots(audio_file, short_term_energy, energy_peaks, [], mfccs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help='Name of the video to process.', metavar='FILE')
    parser.add_argument('--audio', type=str, required=True,
                        help='Name of the video to process.', metavar='FILE')
    args = parser.parse_args()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(f'DEVICE: {device}, VIDEO: {args.video}')

    model = Convolutional(12, 5, 2)
    model.load_state_dict(torch.load('data/models/convolutional.pkl'))

    main(args.video, args.audio, model, device)