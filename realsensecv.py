import pyrealsense2 as rs
import numpy as np


class RealsenseCapture:

    def __init__(self):
        self.WIDTH = 640
        self.HEGIHT = 480
        self.FPS =15
        # Configure depth and color streams
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.WIDTH, self.HEGIHT, rs.format.bgr8, self.FPS)
        self.config.enable_stream(rs.stream.depth, self.WIDTH, self.HEGIHT, rs.format.z16, self.FPS)

    def start(self):
        # Start streaming
        self.pipeline = rs.pipeline()
        # profile = self.pipeline.start(self.config)
        self.pipeline.start(self.config)
        print('pipline start')
        print("\n---------------------\nFPS:", self.FPS, "\n---------------------")
        # https://dev.intelrealsense.com/docs/projection-in-intel-realsense-sdk-20
        # intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        # print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

    def read(self, is_array=True):
        # Flag capture available
        ret = True
        # get frames
        frames = self.pipeline.wait_for_frames()
        # separate RGB and Depth image
        self.color_frame = frames.get_color_frame()  # RGB
        self.depth_frame = frames.get_depth_frame()  # Depth

        if not self.color_frame or not self.depth_frame:
            ret = False
            return ret, (None, None)
        elif is_array:
            # Convert images to numpy arrays
            color_image = np.array(self.color_frame.get_data())
            depth_image = np.array(self.depth_frame.get_data())
            # print('depth: ', self.depth_frame.get_distance(320, 240))
            return ret, (color_image, depth_image), self.depth_frame
        else:
            return ret, (self.color_frame, self.depth_frame), self.depth_frame

    def release(self):
        # Stop streaming
        self.pipeline.stop()
