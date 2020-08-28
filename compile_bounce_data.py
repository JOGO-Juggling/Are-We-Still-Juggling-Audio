from retrieve_utils import BallReader, VideoReader
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

MIN_TRAJ_LEN = 6

def process_ball_trajectory(ball_traj):
    '''Detect bounces in the ball trajectory'''

    # Calculate dy over measured trajectory
    y_trajectory = [ball['y'] for ball in ball_traj if ball != {}]
    dy_trajectory = [y - py for y, py in zip(y_trajectory[:-1], y_trajectory[1:])]

    # Detect if the dy 'goes trough zero'
    change = np.where(np.diff(np.sign(dy_trajectory)))[0]
    return (len(change) > 0 and dy_trajectory[0] < 0)

def extract_ball_data(video_name):
    ball_reader = BallReader('data/balls.json', video_name.split('.')[0])
    video_reader = VideoReader(f'data/videos/{video_name}')
    ms_per_frame = 1000 / video_reader.fps
    ball_trajectory = [{}] * 3

    # Normalize coordinates to 0-1
    x_norm, y_norm = video_reader.shape

    # Resulting data
    result = []
    cur_trajectory = []
    data_len, n_missing = 0, 0

    # Loop over ball data
    for i, ball in enumerate(ball_reader):
        data_len += 1
        ball_trajectory.pop(0)
        
        if ball != {}:
            ball_trajectory.append(ball)
            bounce = process_ball_trajectory(ball_trajectory)

            # Store data in results on new trajectory
            if bounce:
                result.append(cur_trajectory)
                cur_trajectory = []
            cur_trajectory.append((i, i * ms_per_frame, ball['x'] / x_norm, ball['y'] / y_norm))

        else:
            ball_trajectory.append({})
            n_missing += 1
    
    data_quality = 100 - n_missing / data_len * 100
    print('Ball detected in {:.2f}% of frames'.format(data_quality))

    return result, data_quality

class TrajectoryTinder:
    def __init__(self, trajectories):
        self._trajectories = trajectories
        self._done = False
        self._result = []

        self._index = 0
        self._ffi = 0

    def key_press(self, event):
        if event.key == 'q':
            plt.close(event.canvas.figure)
        elif event.key == 'e':
            self.accept()
            if self._index < len(self._trajectories):
                self.get_next()
            plt.close(event.canvas.figure)
        elif event.key == 'w':
            if self._index < len(self._trajectories):
                self.get_next()
            plt.close(event.canvas.figure)
    
    def accept(self):
        self._result.append(self._ffi)
    
    def get_next(self):
        # Get trajectory within requirements
        trajectory = self._trajectories[self._index]
        traj_len = len(trajectory)

        while traj_len < MIN_TRAJ_LEN:
            self._index += 1
            if self._index >= len(self._trajectories):
                return

            trajectory = self._trajectories[self._index]
            traj_len = len(trajectory)

        f, t, x, y = zip(*trajectory)
        self._ffi = f[0]

        t = list(np.array(t) - t[0])

        # Fit curves
        x_fit = np.polyfit(t, x, 1)
        y_fit = np.polyfit(t, y, 2)

        # Plot results
        fig, axs = plt.subplots(1, 2)
        fig.canvas.mpl_connect('key_press_event', self.key_press)
        plt.suptitle('First frame index: {}'.format(f[0]))

        x_line = np.poly1d(x_fit)
        axs[0].plot(t, x, 'o')
        axs[0].plot(t, x_line(t), '--', color='m')
        axs[0].set_title('X trajectory')

        y_line = np.poly1d(y_fit)
        axs[1].plot(t, y, 'o')
        axs[1].plot(t, y_line(t), '--', color='m')
        axs[1].set_title('Y trajectory')

        plt.show()

        # Store results in json
        self._index += 1
    
    def start(self):
        self.get_next()
    
    def get_data(self):
        return self._result

def main(video_name):
    trajectory_data, quality = extract_ball_data(video_name)

    if quality < 85:
        print('Not enough balls detected.')
        return

    trajectory_tinder = TrajectoryTinder(trajectory_data)
    trajectory_tinder.start()

    usable_trajectories = trajectory_tinder.get_data()

    json_data = {}
    with open('data/bounces.json', 'r') as f:
        json_data = json.load(f)

    json_data[video_name] = usable_trajectories
    with open('data/bounces.json', 'w') as f:
        json.dump(json_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help='Name of the video to process.', metavar='FILE')
    args = parser.parse_args()

    print(f'VIDEO: {args.video}')
    main(args.video)