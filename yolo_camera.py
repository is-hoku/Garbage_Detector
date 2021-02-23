from yolo import YOLO
from yolo import detect_video
from timeit import default_timer as timer
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
import numpy as np
import pyrealsense2 as rs
import threading


# RealSenseのtopicをSubscribeしてOpenCVで処理できるようにする．
class RealsenseSubscribe:
    def __init__(self):
        self.frame = 0
        self.depth = 0
        self.frames = []

    def callback(self, msg1, msg2):
        try:
            if (type(msg1) is int) or (type(msg2) is int):
                print("Waiting Frame!!!!!!!")
                pass
            else:
                bridge = CvBridge()
                # print("0")
                orig1 = bridge.imgmsg_to_cv2(msg1, "bgr8")
                orig1 = np.array(orig1, dtype=np.uint8)
                # print("1")
                orig2 = bridge.imgmsg_to_cv2(msg2, 'passthrough')
                orig2 = np.array(orig2, dtype=np.uint8)
                self.frames = [orig1, orig2]
                # cv2.imshow("orig1", orig1)
                # cv2.imshow("orig2", orig2)
                # cv2.waitKey(1)
                # detect_video(YOLO(), frames)

        except Exception as err:
            print(err)

    def read(self):
        # rospy.init_node('img_proc')
        rospy.loginfo('Start to subscribe realsense topic')
        sub1 = message_filters.Subscriber("/camera/color/image_raw", Image)
        sub2 = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
        # rospy.Subscriber("/camera/color/image_raw", Image, self.callback, callback_args=0)
        # rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.callback, callback_args=1)
        fps = 26.77
        delay = 1/fps*0.5
        ts = message_filters.ApproximateTimeSynchronizer([sub1,sub2], 10, delay)
        ts.registerCallback(self.callback)

        rospy.spin()


if __name__ == '__main__':
    video_path = 0
    output_path = './output.avi'
    detect_video(YOLO(), video_path)
    # rospy.init_node('bottle_place')
    # # detect_video(YOLO(), video_path, output_path)
    # vid = RealsenseSubscribe()
    # thread1 = threading.Thread(target=vid.read())
    # thread2 = threading.Thread(target=detect_video(YOLO(),vid.frames))
    # thread1.start()
    # print("thread1")
    # thread2.start()
    # print("thread2")
    # # vid.read()
    # # detect_video(YOLO(), video_path)
