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
    mfccs = librosa.feature.mfcc(signal, samplerate, n_mfcc=40, hop_length=1024).T.tolist()

    false_energies = epd_time.copy()
    m_data = []
    
    ms_per_slice = (duration / len(mfccs)) * 1000
    start_index = (1000 * start) / ms_per_slice

    n_range = [-10, 10]

    prev_bounce = 0

    for bounce in bounce_data:
        closest_epd = min(epd_time, key=lambda x:abs(x - bounce))
        index = int((closest_epd / ms_per_slice) + start_index)
        
        if index < len(mfccs):
            i,j = index + n_range[0], index + n_range[1]
            m = mfccs[i:j]
            s = spectogram[i:j]

            m_data.append({ 'm': m, 't': closest_epd - prev_bounce, 's': 1 })

            if closest_epd in false_energies:
                false_energies.remove(closest_epd)
        prev_bounce = closest_epd
    
    for ms in false_energies:
        b_data = [x for x in bounce_data if x < ms]
        if len(b_data) > 0:
            closest_bounce = min(b_data, key=lambda x:abs(x - ms))
            index = int(ms / ms_per_slice + start_index)

            if index < len(mfccs):
                i,j = index + n_range[0], index + n_range[1]
                m = mfccs[i:j]
                s = spectogram[i:j]

                m_data.append({ 'm': m, 't': ms - closest_bounce, 's': 0 })
    
    mfcc_json_data = {}
    with open('data/mfccs.json', 'r') as f:
        mfcc_json_data = json.load(f)

    mfcc_json_data[videoname] = m_data
    with open('data/mfccs.json', 'w') as f:
        json.dump(mfcc_json_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help='Name of the video to process.', metavar='FILE')
    args = parser.parse_args()

    print(f'VIDEO: {args.video}')
    main('data', args.video)