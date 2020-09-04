import numpy as np
import matplotlib.pyplot as plt

import librosa

from retrieve_utils import VideoReader, AudioReader

from scipy.io import wavfile

import json
import math
import argparse

F_SIZE = 25
F_STRIDE = 10

def main(data_dir, videoname, start=1, stop=11):
    video_path = data_dir + '/video/' + videoname
    audio_path = data_dir + '/audio/' + videoname.split('.')[0] + '.wav'

    videoreader = VideoReader(video_path)
    audioreader = AudioReader(audio_path, F_SIZE, F_STRIDE)

    framerate, samplerate = videoreader.fps, audioreader.samplerate
    duration = videoreader.total_frames / framerate

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

    no_peak_amp = [0]
    ste, ste_time, epd, epd_time = [], [], [], []
    prev_ste0, prev_ste1 = 0, 0

    for i, samples in enumerate(audioreader):
        if i * F_STRIDE > start * 1000:
            if type(samples[0]) is np.ndarray:
                samples = samples.sum(axis=1) / 2 # Convert to 1 channel
            cur_ste = np.mean(np.abs(samples))

            ste.append(cur_ste)
            ste_time.append(i * F_STRIDE)

            mean_amplitude = np.mean(no_peak_amp)
            mean_condition = prev_ste1 > mean_amplitude

            # Energy peak detected
            if prev_ste1 > prev_ste0 and prev_ste1 > cur_ste and mean_condition:
                epd.append(prev_ste1)
                epd_time.append((i - 1) * F_STRIDE)
            else:
                no_peak_amp.append(cur_ste)

            prev_ste0 = prev_ste1
            prev_ste1 = cur_ste

            if i * F_STRIDE > stop * 1000:
                break

    signal, samplerate = librosa.load(audio_path)
    spectogram = librosa.feature.melspectrogram(signal, samplerate)
    spectogram = librosa.power_to_db(spectogram, ref=np.max).T.tolist()
    mfccs = librosa.feature.mfcc(signal, samplerate).T.tolist()

    false_energies = epd_time.copy()
    s_data, m_data = { 'true': [], 'false': [] }, { 'true': [], 'false': [] }
    
    ms_per_slice = (duration / len(spectogram)) * 1000
    start_index = (1000 * start) / ms_per_slice

    n_range = [-5, 10]
    inds = []

    for bounce in bounce_data:
        closest_epd = min(epd_time, key=lambda x:abs(x - bounce))
        index = int((closest_epd / ms_per_slice) + start_index)
        
        if index < len(mfccs):
            inds.append(index)
            i,j = index + n_range[0], index + n_range[1]
            s, m = spectogram[i:j], mfccs[i:j]

            s_data['true'].append(s)
            m_data['true'].append(s)

            if closest_epd in false_energies:
                false_energies.remove(closest_epd)
    
    for ms in false_energies:
        index = int(ms / ms_per_slice + start_index)

        if index < len(mfccs):
            i,j = index + n_range[0], index + n_range[1]
            s, m = spectogram[i:j], mfccs[i:j]
            
            s_data['false'].append(s)
            m_data['false'].append(m)
    
    mfcc_json_data, specto_json_data = {}, {}
    with open('data/specto.json', 'r') as f:
        specto_json_data = json.load(f)    
    with open('data/mfccs.json', 'r') as f:
        mfcc_json_data = json.load(f)

    specto_json_data[videoname] = s_data
    mfcc_json_data[videoname] = m_data
    with open('data/specto.json', 'w') as f:
        json.dump(specto_json_data, f)
    with open('data/mfccs.json', 'w') as f:
        json.dump(mfcc_json_data, f)

    # fig, axs = plt.subplots(3, 1)
    # ste_time, epd_time = np.array(ste_time) / 1000, np.array(epd_time) / 1000

    # # Show STE and detected energy peaks
    # axs[0].plot(ste_time, ste)
    # axs[0].scatter(epd_time, epd, color='r')
    # axs[0].hlines(np.mean(no_peak_amp), xmin=ste_time[0], xmax=ste_time[-1], color='m', linestyles='--')
    # # axs[0].vlines(bounce_data / 1000, ymin=[0] * len(bounce_data), ymax=[max(ste)] * len(bounce_data))
    # axs[0].margins(x=0, y=0.2)
    # axs[0].set_ylabel('Short Time Energy')

    # # Prepare images for display
    # max_frequency = samplerate // 20000
    # spectogram = np.rot90(np.array(spectogram))
    # mfccs = np.rot90(np.array(mfccs))

    # # Show spectogram and peak lines
    # axs[1].vlines(x=inds, ymin=[0] * len(inds), ymax=[100] * len(inds), color='g')
    # axs[1].imshow(spectogram, cmap='jet', aspect="auto")
    # axs[1].set_ylabel('Frequency (kHz)')

    # # Show MFCCs and peak lines
    # axs[2].vlines(x=inds, ymin=[0] * len(inds), ymax=[12] * len(inds), color='g')
    # axs[2].imshow(mfccs[1:13], cmap='jet', aspect="auto")
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