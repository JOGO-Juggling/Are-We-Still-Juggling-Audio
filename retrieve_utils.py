import cv2

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