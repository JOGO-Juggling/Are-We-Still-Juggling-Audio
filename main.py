from retrieve_utils import VideoReader, AudioReader
from draw_utils import draw_frame
from playsound import playsound

import cv2
import numpy as np

from scipy import fftpack
from time import sleep

def main():
    cv2.namedWindow('Juggling', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Juggling', 600, 600)


    videoreader = VideoReader('data/videos/133de14e96374d0d9110e782587c170d.MOV')
    audioreader = AudioReader('data/audio/133de14e96374d0d9110e782587c170d.wav')
    
    frame_w, frame_h = videoreader.shape
    framerate = videoreader.fps
    audiorate = audioreader.samplerate

    for frame, audio in zip(videoreader, audioreader):
        n = len(audio)
        audio = [int(x[0]) for x in audio]
        fourier = fftpack.fft(audio)
        spectrum = np.abs(fourier[70:n//2 - 300])
        # spectrum = np.abs(fourier[:n//2])

        bins = 25
        bin_size = len(spectrum) // bins
        binned = [np.mean(spectrum[i * bin_size:i * bin_size + bin_size]) for i in range(bins)]

        frame = frame[1]
        frame = draw_frame(frame, binned[3:], frame_w, frame_h)
        cv2.imshow('Juggling', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sleep(int(1000 / framerate) / 500)
main()