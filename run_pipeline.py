import torch
import argparse
import librosa
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from retrieve_utils import AudioReader, VideoReader
from models import Convolutional

N_FFT = 4096
N_MFCC = 40

F_RANGE = (2, 40)
W_RANGE = (-3, 2)
W_CENTER = 10

n_features = abs(F_RANGE[1] - F_RANGE[0])
win_dim = abs(W_RANGE[1] - W_RANGE[0])

def get_energy_peaks(v_file, a_file, f_size=25, f_stride=10):
    '''Calculates the energy peaks for an audio file, and returns the timings
       (in ms) of the energy peaks detected. Returns a list of EP timings.'''
    audioreader = AudioReader(a_file, f_size, f_stride)
    videoreader = VideoReader(v_file)

    total_frames, framerate = videoreader.total_frames, videoreader.fps

    no_peak_amp, ste, epd = [0], [], []
    prev_ste0, prev_ste1 = 0, 0

    stop = int(total_frames / framerate) - 1

    for i, samples in enumerate(audioreader):
        if type(samples[0]) is np.ndarray:
            samples = samples.sum(axis=1) / 2 # Convert to 1 channel
        cur_ste = np.mean(np.abs(samples))
        ste.append(cur_ste)

        # Filter out all energy peaks < mean_amplitude of non-peaks
        mean_amplitude = np.mean(no_peak_amp)
        mean_condition = prev_ste1 > mean_amplitude

        if prev_ste1 > prev_ste0 and prev_ste1 > cur_ste and mean_condition:
            epd.append((i - 1) * f_stride)
        else:
            no_peak_amp.append(cur_ste) # Update non-peak amplitudes

        prev_ste0 = prev_ste1
        prev_ste1 = cur_ste

        # Stop 1 second before end of video
        if i * f_stride > stop * 1000:
            break
    
    return ste, epd


def get_mfccs(a_file, ep_timings):
    '''Constructs the MFCC feature matrix for each energy peak in an audio file
       given the timings of the energy peaks. Returns a list of MFCC matrices'''
    duration = librosa.get_duration(filename=a_file)
    signal, samplerate = librosa.load(a_file)
    mfccs = librosa.feature.mfcc(signal, samplerate, n_mfcc=N_MFCC, n_fft=N_FFT).T
    ms_per_slice = (duration / len(mfccs)) * 1000
    epd_mfccs = []

    w_range = (W_RANGE[0] + W_CENTER, W_CENTER + W_RANGE[1])

    # Iterate trough energy peaks, match 
    for ep in ep_timings:
        mfcc_index = int((ep / ms_per_slice))

        if mfcc_index < len(mfccs) - 1:
            i,j = mfcc_index + w_range[0], mfcc_index + w_range[1]
            matrix = mfccs[i:j,F_RANGE[0]:F_RANGE[1]]
            epd_mfccs.append(np.abs([matrix]))
    return np.array(epd_mfccs), mfccs


def display_plots(ste, true_epd, false_epd, mfccs):
    '''Displays the results of the extraction functions in a plot.'''
    fig, axs = plt.subplots(2, 1)
    ste_time = np.arange(len(ste)) * 10

    # Plot STE and detected energy peaks
    axs[0].plot(ste_time, ste, color='b')
    axs[0].vlines(true_epd, [0] * len(true_epd), [max(ste)] * len(true_epd),
               color='g', linestyles='--')
    axs[0].vlines(false_epd, [0] * len(false_epd), [max(ste)] * len(false_epd),
               color='r', linestyles='--')
    axs[1].imshow(np.rot90(mfccs[:,2:14]))
    plt.show()


def render_frame(frame, bounce):
    '''Adds bounce text to frame if bounce is detected and returns frame.'''
    if bounce:
        frame = cv2.putText(frame, 'BOUNCE', (20, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0))
    return frame


def run_result(v_file, predictions, ep_timings):
    '''Loops over each frame in the video and displays the predicted bounces.'''
    videoreader = VideoReader(v_file)
    framerate = videoreader.fps
    ms_per_frame = 1000 / framerate
    cur_ep_index = 0

    for i, frame in enumerate(videoreader):
        bounce = False

        # Energy was peak detected in current frame
        if i * ms_per_frame >= ep_timings[cur_ep_index]:
            if predictions[cur_ep_index] == 1:
                bounce = True
            cur_ep_index += 1
        
        frame = render_frame(frame[1], bounce)
        cv2.imshow('Result', frame)

        if cv2.waitKey(int(ms_per_frame) * 2) & 0xFF == ord('q'):
            break


def make_predictions(model, mfccs):
    with torch.no_grad():
        y_hat = model(mfccs.to(device)).cpu()
    _, predictions = torch.max(y_hat.data, 1)
    return predictions.tolist()


def main(video_file, audio_file, model, device):
    # Extract energy peaks
    short_term_energy, energy_peaks = get_energy_peaks(video_file, audio_file)

    # Extract MFCCs
    epd_mfccs, mfccs = get_mfccs(audio_file, energy_peaks)
    mfccs_batch = torch.FloatTensor(epd_mfccs)

    # Prepare model
    model.eval()
    model.to(device)

    # For each energy peak make prediction using MFCCs
    print(mfccs_batch.shape)
    predictions = make_predictions(model, mfccs_batch)
    true_peaks = [x for x, y in zip(energy_peaks, predictions) if y == 1]
    false_peaks = [x for x, y in zip(energy_peaks, predictions) if y == 0]

    # Display predictions
    display_plots(short_term_energy, true_peaks, false_peaks, mfccs)
    run_result(video_file, predictions, energy_peaks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help='Name of the video to process.', metavar='FILE')
    parser.add_argument('--audio', type=str, required=True,
                        help='Name of the video to process.', metavar='FILE')
    args = parser.parse_args()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(f'DEVICE: {device}, VIDEO: {args.video}')

    model = Convolutional(n_features, win_dim, 2)
    model.load_state_dict(torch.load('data/models/convolutional.pkl'))

    main(args.video, args.audio, model, device)