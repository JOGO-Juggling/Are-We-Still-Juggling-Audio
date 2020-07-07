from retrieve_utils import VideoReader
from draw_utils import draw_frame

import cv2
import numpy as np

def main():
    cv2.namedWindow('Juggling', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Juggling', 600, 600)

    video_path = 'data/videos/133de14e96374d0d9110e782587c170d.MOV'
    videoreader = VideoReader(video_path)
    frame_w, frame_h = videoreader.shape

    for frame in videoreader:
        random_data = list(np.random.choice(10, 100))

        frame = frame[1]
        frame = draw_frame(frame, random_data, frame_w, frame_h)
        cv2.imshow('Juggling', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()