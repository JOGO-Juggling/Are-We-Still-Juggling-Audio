import scipy.io
import numpy as np

from os.path import dirname, join as pjoin
from scipy.io import wavfile

class FrameIterator: 
    def __init__(self, data_path):
        self.samplerate, self.data = wavfile.read(data_path)
        self.length = self.data.shape[0] / self.samplerate
        self.time = np.linspace(0., self.length, self.data.shape[0])
        self.frame = round(self.samplerate / self.length)
        self.cur_sample = 0

    def __iter__(self):
        # Create iterator
        self.cur_sample = 0
        return self

    def __next__(self):
        # Loop over frames
        if self.cur_sample < self.samplerate:
            result = self.data[self.cur_sample : self.cur_sample + self.frame]
            self.cur_sample += self.frame
            return result
        else:
            raise StopIteration

# frame = FrameIterator('./data/133de14e96374d0d9110e782587c170d.wav')

# for i, f in enumerate(frame):
#     print(np.mean(f))
#     if i > 180:
#         break

