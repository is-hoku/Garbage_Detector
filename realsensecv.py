import pyrealsense2 as rs
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
import message_filters


# RealSenseのtopicを利用せず，直接コンピュータにUSBで接続して使う場合は以下をコメントアウト

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

    def call_back(ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)

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
            # print("color_image = ", color_image)
            depth_image = np.array(self.depth_frame.get_data())
            # print("depth_image = ", depth_image)
            # print('depth: ', self.depth_frame.get_distance(320, 240))
            return ret, (color_image, depth_image), self.depth_frame
        else:
            return ret, (self.color_frame, self.depth_frame), self.depth_frame

    def release(self):
        # Stop streaming
        self.pipeline.stop()



# RealSenseのtopicをSubscribeしてOpenCVで処理できるようにする．
    # def __init__(self):
    #     self.frame = 0
    #     self.depth = 0
    #     # self.cnt = 0
    #     # self.fps = 0.
    #     # self.start = 0
    #     # self.end = 0
    #
    # # def callback(self, msg, flag):
    # #     try:
    # #         bridge = CvBridge()
    # #         if flag == 0:
    # #             # if self.cnt == 0:
    # #             #     self.start = timer()
    # #             print("0")
    # #             orig = bridge.imgmsg_to_cv2(msg, "bgr8")
    # #             # print(type(msg))
    # #             # img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # #             # print(orig)
    # #             # print(orig.shape)
    # #             self.frame = orig
    # #             # cv2.imshow('image', orig)
    # #             # cv2.waitKey(1)
    # #         else:
    # #             # if self.cnt == 10:
    # #             #     self.end = timer()
    # #             #     self.fps = self.end - self.start
    # #             # self.cnt += 1
    # #
    # #             print("1")
    # #             orig = bridge.imgmsg_to_cv2(msg, 'passthrough')
    # #             # cv2.imshow('image2', orig)
    # #             # cv2.waitKey(1)
    # #             self.depth = orig
    # #     except Exception as err:
    # #         print(err)
    #
    # def callback(self, msg1, msg2):
    #     try:
    #         bridge = CvBridge()
    #         print("0")
    #         orig1 = bridge.imgmsg_to_cv2(msg1, "bgr8")
    #         print("1")
    #         orig2 = bridge.imgmsg_to_cv2(msg2, 'passthrough')
    #         frame = [orig1, orig2]
    #         # cv2.imshow("orig1", orig1)
    #         # cv2.imshow("orig2", orig2)
    #         # cv2.waitKey(1)
    #
    #     except Exception as err:
    #         print(err)
    #
    # def read(self):
    #     rospy.init_node('img_proc')
    #     rospy.loginfo('Start to subscribe realsense topic')
    #     sub1 = message_filters.Subscriber("/camera/color/image_raw", Image)
    #     sub2 = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
    #     # rospy.Subscriber("/camera/color/image_raw", Image, self.callback, callback_args=0)
    #     # rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.callback, callback_args=1)
    #     fps = 26.77
    #     delay = 1/fps*0.5
    #     ts = message_filters.ApproximateTimeSynchronizer([sub1,sub2], 10, delay)
    #     ts.registerCallback(self.callback)
    #
    #     rospy.spin()


# start_node()

# vid = RealsenseCapture()

# for i in range(200):

# vid.start()
# vid.read()

# print(vid.fps)
    # print(vid.frame.shape)
    # print(vid.frame)
    # cv2.imshow('frame', vid.frame)
    # cv2.waitKey(1)


# cv2.imshow("frame", frame)
