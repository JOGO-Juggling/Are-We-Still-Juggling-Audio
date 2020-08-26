import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy import fftpack
import math

class AudioReader: 
    def __init__(self, data_path, frame_size, frame_stride):
        self.samplerate, self.data = wavfile.read(data_path)
        self.length = self.data.shape[0]
        self.frame_size = int((self.samplerate / 1000) * frame_size)
        self.frame_stride = int((self.samplerate / 1000) * frame_stride)

    def __iter__(self):
        # Create iterator
        self.cur_sample = 0
        return self

    def __next__(self):
        # Loop over frames
        if self.cur_sample < self.length:
            result = self.data[self.cur_sample:self.cur_sample + self.frame_size]
            self.cur_sample += self.frame_stride
            return result
        else:
            raise StopIteration

def main(audio_path):
    frame_size, frame_stride = 30, 10
    audioreader = AudioReader(audio_path, frame_size, frame_stride)
    wav_output = []

    ste, ste_time, epd, epd_time = [], [], [], []
    prev_ste0, prev_ste1 = 0, 0

    for i, samples in enumerate(audioreader):
        cur_ste = np.mean(np.abs(samples))

        # if i * frame_stride > 1000:
        wav_output += list(samples)
        if prev_ste1 > prev_ste0 and prev_ste1 > cur_ste:
            if abs(prev_ste0 - prev_ste1) > 5 and cur_ste > 400:
                epd.append(prev_ste1)
                epd_time.append((i - 1) * frame_stride)

        ste.append(cur_ste)
        ste_time.append(i * frame_stride)

        prev_ste0 = prev_ste1
        prev_ste1 = cur_ste

        # if i * frame_stride > 35000:
        #     break
    
    print(len(epd))
    wavfile.write('out.wav', audioreader.samplerate * 3, np.array(wav_output))
    plt.plot(ste_time, ste)
    plt.scatter(epd_time, epd, color='r')
    plt.show()

if __name__ == '__main__':
    main('data/audio/7fc0c7ecc2394668a9b43e6724a46770.wav')