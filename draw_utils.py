import cv2
import numpy as np

def bar_plot(frame, data, width, height, position):
    max_data = np.max(data)

    if max_data > 0:
        data /= max_data
    data *= height

    x0, y0 = position
    x1 = x0 + width

    n_bars = len(data)
    bar_width = width // n_bars
    x_data = np.linspace(x0, x1 - bar_width, n_bars)

    for x, y in zip(x_data, data):
        frame = cv2.rectangle(frame, (int(x), int(y0 - y)), (int(x) + bar_width, y0), (255, 0, 0))
    
    return frame

def draw_frame(frame, audio, width, height):
    return bar_plot(frame, audio, width, height // 5, (0, height))