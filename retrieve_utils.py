import scipy.io
import numpy as np

from os.path import dirname, join as pjoin
from scipy.io import wavfile

import cv2

class AudioReader: 
    def __init__(self, data_path):
        self.samplerate, self.data = wavfile.read(data_path)
        self.length = self.data.shape[0]
        self.time = self.length / self.samplerate
        self.frame = round(self.samplerate / self.time)

    def __iter__(self):
        # Create iterator
        self.cur_sample = 0
        return self

    def __next__(self):
        # Loop over frames
        if self.cur_sample < self.length:
            result = self.data[self.cur_sample : self.cur_sample + self.frame]
            self.cur_sample += self.frame
            return result
        else:
            raise StopIteration

class VideoReader:
    def __init__(self, video_path):

        self.video = cv2.VideoCapture(video_path)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.shape = (int(width), int(height))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        self.cur_frame = 0
        return self

    def __next__(self):
        if self.cur_frame < self.total_frames:
            self.cur_frame += 1

            return self.video.read()
        else:
            raise StopIteration