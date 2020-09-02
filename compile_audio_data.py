import numpy as np
import matplotlib.pyplot as plt

from retrieve_utils import VideoReader, AudioReader

from scipy.fftpack import dct
from scipy.io import wavfile

import json
import math
import argparse

N_FFT = 512
N_FIL = 40
N_CEP = 12

F_SIZE = 50
F_STRIDE = 25

def construct_slice(samples, samplerate, frame_size):
    # Applt hamming frame
    samples = samples * np.hamming((frame_size / 1000) * samplerate)

    # Apply FFT and calculate power spectrum
    mag_spectrum = np.abs(np.fft.rfft(samples, N_FFT))
    pow_spectrum = ((1.0 / N_FFT) * ((mag_spectrum) ** 2))

    # Convert to mel scale and back
    max_mel_frequency = (2595 * np.log10(1 + (samplerate / 2) / 700)) # Hz to Mel
    mel_points = np.linspace(0, max_mel_frequency, N_FIL + 2) # Create normalisation points
    hez_points = (700 * (10 ** (mel_points / 2595) - 1)) # Covernt Mel back to Hz

    # Create empty filterbank
    bands = np.floor((N_FFT + 1) * hez_points / samplerate)
    filter_bank = np.zeros((N_FIL, int(np.floor(N_FFT / 2 + 1))))

    # Loop over filterbank
    for f in range(1, N_FIL + 1):
        # Get left, center and right
        l_band, c_band, r_band = int(bands[f - 1]), int(bands[f]), int(bands[f + 1])

        # Fill filterbank
        for i in range(l_band, c_band):
            filter_bank[f - 1, i] = (i - bands[f - 1]) / (bands[f] - bands[f - 1])
        for i in range(c_band, r_band):
            filter_bank[f - 1, i] = (bands[f + 1] - i) / (bands[f + 1] - bands[f])
    
    # Apply filterbank to signal
    filtered = np.dot(pow_spectrum, filter_bank.T)
    filtered = np.where(filtered == 0, np.finfo(float).eps, filtered)
    return 20 * np.log10(filtered)

def get_MFCCs(periodogram_slice):
    # Apply discrete cosine transform and keep coefficients 2-13
    return dct(periodogram_slice, type=2, axis=0, norm='ortho')[2:(N_CEP + 2)]

def main(data_dir, videoname, start=1, stop=11):
    video_path = data_dir + '/video/' + videoname
    audio_path = data_dir + '/audio/' + videoname.split('.')[0] + '.wav'

    videoreader = VideoReader(video_path)
    audioreader = AudioReader(audio_path, F_SIZE, F_STRIDE)

    framerate, samplerate = videoreader.fps, audioreader.samplerate
    stop = int(videoreader.total_frames / framerate) - 1

    # Get all detected bounces
    bounce_data = []
    with open(data_dir + '/bounces.json') as file:
        json_data = json.load(file)
        if videoname not in json_data:
            print('Video not in bounces.json')
            return
        bounce_data = json_data[videoname]

    # Covert video frame indices to ms
    bounce_data = (np.array(bounce_data) / framerate) * 1000
    bounce_data = bounce_data[bounce_data > start * 1000]
    bounce_data = bounce_data[bounce_data < stop * 1000]

    periodogram, mfccs = [], []

    no_peak_amp = [0]
    ste, ste_time, epd, epd_time = [], [], [], []
    prev_ste0, prev_ste1 = 0, 0

    for i, samples in enumerate(audioreader):
        if i * F_STRIDE > start * 1000:
            if type(samples) is np.ndarray:
                samples = samples.sum(axis=1) / 2 # Covert to 1 channel
            cur_ste = np.mean(np.abs(samples))

            mean_amplitude = np.mean(no_peak_amp)
            mean_condition = prev_ste1 > mean_amplitude - 0.05 * mean_amplitude

            # Energy peak detected
            if prev_ste1 > prev_ste0 and prev_ste1 > cur_ste and mean_condition:
                epd.append(prev_ste1)
                epd_time.append((i - 1) * F_STRIDE)

                # Construct periodogram
                periodogram_slice = construct_slice(samples, samplerate, F_SIZE)
                periodogram.append(list(periodogram_slice.T))

                # Get MFCCs
                mfccs.append(list(get_MFCCs(periodogram_slice).T))
            else:
                # periodogram.append(np.zeros(40,).T)
                # mfccs.append(np.zeros(N_CEP,).T)
                no_peak_amp.append(cur_ste)

            ste.append(cur_ste)
            ste_time.append(i * F_STRIDE)

            prev_ste0 = prev_ste1
            prev_ste1 = cur_ste

            if i * F_STRIDE > stop * 1000:
                break
    
    perio_copy, mfccs_copy = periodogram.copy(), mfccs.copy()
    perio_dataset, mfcc_dataset = { 'true': [], 'false': [] }, { 'true': [], 'false': [] }

    for bounce in bounce_data:
        closest = min(epd_time, key=lambda x:abs(x - bounce))
        closest_ind = epd_time.index(closest)

        if closest < bounce and closest_ind < len(epd_time):
            closest_ind += 1
        
        if closest_ind < len(epd_time):
            perio = periodogram[closest_ind]
            mfcc = mfccs[closest_ind]

            perio_dataset['true'].append(perio)
            mfcc_dataset['true'].append(mfcc)

            if perio in perio_copy:
                perio_copy.remove(perio)
            if mfcc in mfccs_copy:
                mfccs_copy.remove(mfcc)

    for perio in perio_copy:
        perio_dataset['false'].append(perio)
    for mfcc in mfccs_copy:
        mfcc_dataset['false'].append(mfcc)
    
    mfcc_json_data, perio_json_data = {}, {}
    with open('data/perio.json', 'r') as f:
        perio_json_data = json.load(f)    
    with open('data/mfccs.json', 'r') as f:
        mfcc_json_data = json.load(f)

    perio_json_data[videoname] = perio_dataset
    mfcc_json_data[videoname] = mfcc_dataset
    with open('data/perio.json', 'w') as f:
        json.dump(perio_json_data, f)
    with open('data/mfccs.json', 'w') as f:
        json.dump(mfcc_json_data, f)

    fig, axs = plt.subplots(3, 1)
    ste_time, epd_time = np.array(ste_time) / 1000, np.array(epd_time) / 1000

    # Show STE and detected energy peaks
    # axs[0].plot(ste_time, ste)
    # axs[0].scatter(epd_time, epd, color='r')
    # axs[0].hlines(np.mean(no_peak_amp), xmin=ste_time[0], xmax=ste_time[-1], color='m', linestyles='--')
    # # axs[0].vlines(bounce_data / 1000, ymin=[0] * len(bounce_data), ymax=[max(ste)] * len(bounce_data))
    # axs[0].margins(x=0, y=0.2)
    # axs[0].set_ylabel('Short Time Energy')

    # # Prepare images for display
    # max_frequency = samplerate // 20000
    # periodogram = np.rot90(np.array(periodogram))
    # mfccs = np.rot90(np.array(mfccs))

    # # Show periodogram and peak lines
    # axs[1].imshow(periodogram, cmap='jet', extent=[start, stop, 0, max_frequency], aspect="auto")
    # axs[1].set_ylabel('Frequency (kHz)')

    # # Show MFCCs and peak lines
    # axs[2].imshow(mfccs, cmap='jet', extent=[start, stop, 0, N_CEP], aspect="auto")
    # axs[2].set_ylabel('MFCC Values')
    # axs[2].set_xlabel('Time (s)')

    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help='Name of the video to process.', metavar='FILE')
    args = parser.parse_args()

    print(f'VIDEO: {args.video}')
    main('data', args.video)