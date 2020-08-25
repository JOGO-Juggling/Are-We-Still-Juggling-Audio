import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy import fftpack
import math

class AudioReader: 
    def __init__(self, data_path, window_size, interval):
        self.samplerate, self.data = wavfile.read(data_path)
        self.length = self.data.shape[0]
        self.window_size = int((self.samplerate / 1000) * window_size)
        self.interval = int((self.samplerate / 1000) * interval)

    def __iter__(self):
        # Create iterator
        self.cur_sample = 0
        return self

    def __next__(self):
        # Loop over frames
        if self.cur_sample < self.length:
            result = self.data[self.cur_sample:self.cur_sample + self.window_size]
            self.cur_sample += self.interval
            return result
        else:
            raise StopIteration

def main(audio_path):
    audioreader = AudioReader(audio_path, 150, 50)
    peak_variance = 100

    ste, epd, epd_val, avg_peak = [0], [], [], []

    for i, samples in enumerate(audioreader):
        cur_ste = np.mean(np.abs(samples))
        angle = math.degrees(math.atan(cur_ste - ste[-1]))

        peak_avg = np.mean(avg_peak)
        within_avg = cur_ste > peak_avg - peak_variance and cur_ste < peak_avg + peak_variance and peak_avg

        if len(avg_peak) < 1:
            within_avg = True

        if i > 50:
            prev_epd = epd[-1] if len(epd) > 0 else -10
            if angle > 80 and within_avg and i - prev_epd > 4:
                epd.append(i)
                epd_val.append(cur_ste)
                avg_peak.append(cur_ste)
                # if len(avg_peak) > 10:
                #     avg_peak.pop(0)
        ste.append(cur_ste)

        if i > 500:
            break
    
    ste = np.array(ste)
    plt.plot(np.arange(len(ste)), ste)
    plt.hlines(np.mean(avg_peak), xmin=0, xmax=len(ste), linestyle='--', color='m')
    plt.hlines(np.mean(avg_peak) + peak_variance, xmin=0, xmax=len(ste), linestyle='-.', color='g')
    plt.hlines(np.mean(avg_peak) - peak_variance, xmin=0, xmax=len(ste), linestyle='-.', color='g')
    plt.scatter(epd, epd_val, color='r')

    plt.show()

if __name__ == '__main__':
    main('data/audio/IMG_0786.wav')