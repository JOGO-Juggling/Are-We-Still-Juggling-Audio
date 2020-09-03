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

class BounceTinder:
    def __init__(self, energy_peaks, short_term_energy, spectogram, videoreader):
        self._epd, self._epd_time = energy_peaks
        self._ste, self._ste_time = short_term_energy
        self._spectogram = spectogram
        self._result = []

        self._frame_index, self._epd_index = 0, 0
        self._n_frames, self._n_epds = videoreader.total_frames, len(self._epd)
        self._framerate = videoreader.fps
        self._videoreader = iter(videoreader)
        self._frame_stack = [None, None, None]
    
    def key_press(self, event):
        if event.key == 'q':
            plt.close(event.canvas.figure)
        elif event.key == 'e':
            self.accept()
            if self._frame_index < self._n_frames and self._epd_index < self._n_epds:
                self.get_next()
            plt.close(event.canvas.figure)
        elif event.key == 'w':
            if self._frame_index < self._n_frames and self._epd_index < self._n_epds:
                self.get_next()
            plt.close(event.canvas.figure)
    
    def accept(self):
        self._result.append(self._frame_index)
    
    def next_frame(self):
        frame = next(self._videoreader)
        self._frame_stack.append(frame)
        self._frame_stack.pop(0)
    
    def get_next(self):
        self._frame_index += 1
        if self._frame_index >= self._n_frames:
            return
        self.next_frame()

        # Find frame just before bounce
        while self._epd_time[self._epd_index] > (self._frame_index / self._framerate) * 1000:
            self._frame_index += 1
            if self._frame_index >= self._n_frames:
                return
            self.next_frame()

        # Select part of STE to show
        ste_start_el = [x for x in self._ste_time if x > self._epd_time[self._epd_index]][0]
        ste_start = self._ste_time.index(ste_start_el) - 20
        ste_stop = self._ste_time.index(ste_start_el) + 20

        if ste_start < 0:
            ste_start = 0
        if ste_stop >= len(self._ste_time):
            ste_stop = len(self._ste_time) - 1
        
        fig = plt.figure(figsize=(15, 15))
        fig.canvas.mpl_connect('key_press_event', self.key_press)


        plt.subplot(231)
        plt.imshow(self._frame_stack[0][1][...,::-1])
        plt.subplot(232)
        plt.imshow(self._frame_stack[1][1][...,::-1])
        plt.subplot(233)
        plt.imshow(self._frame_stack[2][1][...,::-1])

        plt.subplot(212)
        plt.plot(self._ste_time[ste_start:ste_stop], self._ste[ste_start:ste_stop])
        plt.plot(self._epd_time[self._epd_index], self._epd[self._epd_index], 'o')
        plt.show()

        self._epd_index += 1
    
    def start(self):
        self.get_next()
    
    def get_data(self):
        return self._result

def main(data_dir, videoname, start=1, stop=20):
    video_path = data_dir + '/video/' + videoname
    audio_path = data_dir + '/audio/' + videoname.split('.')[0] + '.wav'

    videoreader = VideoReader(video_path)
    audioreader = AudioReader(audio_path, F_SIZE, F_STRIDE)

    framerate, samplerate = videoreader.fps, audioreader.samplerate

    stop = int(videoreader.total_frames / framerate) - 1

    no_peak_amp = [0]
    ste, ste_time, epd, epd_time = [], [], [], []
    prev_ste0, prev_ste1 = 0, 0

    for i, samples in enumerate(audioreader):
        if i * F_STRIDE > start * 1000:
            if type(samples[0]) is np.ndarray:
                samples = samples.sum(axis=1) / 2 # Covert to 1 channel
            cur_ste = np.mean(np.abs(samples))

            ste.append(cur_ste)
            ste_time.append(i * F_STRIDE)

            mean_amplitude = np.mean(no_peak_amp)
            mean_condition = prev_ste1 > mean_amplitude - 0.05 * mean_amplitude

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
    
    print(f'{len(epd)} energy peaks detected! Veel plezier..')
    bounce_tinder = BounceTinder((epd, epd_time), (ste, ste_time), spectogram, videoreader)
    bounce_tinder.start()

    bounce_data = bounce_tinder.get_data()
    json_data = []
    with open('data/bounces.json', 'r') as f:
        json_data = json.load(f)

    json_data[videoname] = bounce_data
    with open('data/bounces.json', 'w') as f:
        json.dump(json_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help='Name of the video to process.', metavar='FILE')
    args = parser.parse_args()

    print(f'VIDEO: {args.video}')
    main('data', args.video)