import torch
import argparse
import librosa
import numpy as np

from retrieve_utils import AudioReader, VideoReader
from models import Convolutional

def get_energy_peaks(v_file, a_file, f_size=25, f_stride=10):
    '''Calculates the energy peaks for an audio file, and returns the timings
       (in ms) of the energy peaks detected. Returns a list of EP timings.'''
    audioreader = AudioReader(a_file, f_size, f_stride)
    videoreader = VideoReader(v_file)

    total_frames, framerate = videoreader.total_frames, videoreader.fps

    no_peak_amp, result = [0], []
    prev_ste0, prev_ste1 = 0, 0

    stop = int(total_frames / framerate) - 1

    for i, samples in enumerate(audioreader):
        if type(samples[0]) is np.ndarray:
            samples = samples.sum(axis=1) / 2 # Convert to 1 channel
        ste = np.mean(np.abs(samples))

        # Filter out all energy peaks < mean_amplitude of non-peaks
        mean_amplitude = np.mean(no_peak_amp)
        mean_condition = prev_ste1 > mean_amplitude

        if prev_ste1 > prev_ste0 and prev_ste1 > ste and mean_condition:
            result.append(i * f_stride)
        else:
            no_peak_amp.append(ste) # Update non-peak amplitudes

        prev_ste0 = prev_ste1
        prev_ste1 = ste

        # Stop 1 second before end of video
        if i * f_stride > stop * 1000:
            break
    
    return result


def get_mfccs(a_file, ep_timings, n_range=[-2, 3]):
    '''Constructs the MFCC feature matrix for each energy peak in an audio file
       given the timings of the energy peaks. Returns a list of MFCC matrices'''
    duration = librosa.get_duration(filename=file)
    signal, samplerate = librosa.load(file)
    mfccs = librosa.feature.mfcc(signal, samplerate).T.tolist()
    ms_per_slice = (duration / len(spectogram)) * 1000

    result = []

    # Iterate trough energy peaks, match 
    for ep in ep_timings:
        mfcc_index = int((ep / ms_per_slice))

        if mfcc_index < len(mfccs) - 1:
            i,j = index + n_range[0], index + n_range[1]
            result.append([mfcss[i:j]])
    
    return result

def show_result(v_file, predictions, ep_timings):
    '''Loops over each frame in the video and displays the predicted bounces.'''
    videoreader = VideoReader(file)
    framerate = videoreader.fps
    ms_per_frame = 1000 / framerate
    cur_ep_index = 0

    for i, frame in enumerate(videoreader):
        # Energy was peak detected in current frame
        if i * ms_per_frame >= ep_timings[cur_ep_index]:
            cur_ep_index += 1

def main(video_file, audio_file):
    # Extract energy peaks
    energy_peaks = get_energy_peaks(video_file, audio_file)

    # Extract MFCCs
    mfccs = get_mfccs(audio_file, energy_peaks)
    mfccs_batch = torch.FloatTensor(mfccs)

    # For each energy peak make prediction using MFCCs
    model = Convolutional(12, 5, 2).eval()
    model.load_state_dict('data/models/convolutional.pkl')

    with torch.no_grad():
        _, predictions = torch.max(model(mfccs_batch).data, 1)
    print(predictions)

    # Display predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help='Name of the video to process.', metavar='FILE')
    parser.add_argument('--audio', type=str, required=True,
                        help='Name of the video to process.', metavar='FILE')
    args = parser.parse_args()

    print(f'VIDEO: {args.video}')
    main(args.video, args.audio)