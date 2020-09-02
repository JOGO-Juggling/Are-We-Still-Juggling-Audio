import numpy as np
import matplotlib.pyplot as plt

from retrieve_utils import VideoReader, AudioReader

from scipy.io import wavfile

import json
import math
import argparse

N_FFT = 512
N_FIL = 40

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


class BounceTinder:
    def __init__(self, energy_peaks, short_term_energy, periodogram, videoreader):
        self._epd, self._epd_time = energy_peaks
        self._ste, self._ste_time = short_term_energy
        self._periodogram = periodogram
        self._done = False
        self._result = []

        self._index = 0
        self._epd_index = 0
        self._framerate = videoreader.fps
        self._total_frames = videoreader.total_frames
        self._videoreader = iter(videoreader)
    
    def key_press(self, event):
        if event.key == 'q':
            plt.close(event.canvas.figure)
        elif event.key == 'e':
            self.accept()
            if self._index < self._total_frames and self._epd_index < len(self._epd):
                self.get_next()
            plt.close(event.canvas.figure)
        elif event.key == 'w':
            if self._index < self._total_frames and self._epd_index < len(self._epd):
                self.get_next()
            plt.close(event.canvas.figure)
    
    def accept(self):
        self._result.append(self._index - 1)
    
    def get_next(self):
        self._index +=1
        if self._index >= self._total_frames:
            return
        frame = next(self._videoreader)
        prev_frame = frame

        # Find frame just before bounce
        while self._epd_time[self._epd_index] > (self._index / self._framerate) * 1000:
            if self._index >= self._total_frames:
                return
            prev_frame = frame
            frame = next(self._videoreader)
            self._index += 1

        ste_start_el = [x for x in self._ste_time if x > self._epd_time[self._epd_index]][0]
        ste_start = self._ste_time.index(ste_start_el) - 20
        ste_stop = self._ste_time.index(ste_start_el) + 20

        if ste_start < 0:
            ste_start = 0
        if ste_stop >= len(self._ste_time):
            ste_stop = len(self._ste_time) - 1
        
        fig = plt.figure(figsize=(15, 15))
        fig.canvas.mpl_connect('key_press_event', self.key_press)

        plt.subplot(3, 1, 1)
        plt.imshow(prev_frame[1][...,::-1])

        plt.subplot(3, 1, 2)
        plt.plot(self._ste_time[ste_start:ste_stop], self._ste[ste_start:ste_stop])
        plt.plot(self._epd_time[self._epd_index], self._epd[self._epd_index], 'o')

        plt.subplot(3, 1, 3)
        plt.imshow(np.rot90(np.array(self._periodogram[ste_start:ste_stop])), cmap='jet')
        plt.show()

        self._epd_index += 1
    
    def start(self):
        self.get_next()
    
    def get_data(self):
        return self._result

def main(data_dir, videoname, start=5, stop=20):
    video_path = data_dir + '/videos/' + videoname
    audio_path = data_dir + '/audio/' + videoname.split('.')[0] + '.wav'

    videoreader = VideoReader(video_path)
    audioreader = AudioReader(audio_path, F_SIZE, F_STRIDE)

    framerate, samplerate = videoreader.fps, audioreader.samplerate

    stop = int(videoreader.total_frames / framerate) - 1

    no_peak_amp = [0]
    ste, ste_time, epd, epd_time = [], [], [], []
    periodogram = []
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

            # Construct periodogram
            periodogram_slice = construct_slice(samples, samplerate, F_SIZE)
            periodogram.append(list(periodogram_slice.T))

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

    # fig, axs = plt.subplots(2, 1)
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
    # periodogram = np.rot90(np.array(periodogram))
    # # mfccs = np.rot90(np.array(mfccs))

    # # Show periodogram and peak lines
    # axs[1].imshow(periodogram, cmap='jet', extent=[start, stop, 0, max_frequency], aspect="auto")
    # axs[1].set_ylabel('Frequency (kHz)')
    # plt.show()
    
    print(f'{len(epd)} energy peaks detected! Veel plezier..')
    bounce_tinder = BounceTinder((epd, epd_time), (ste, ste_time), periodogram, videoreader)
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