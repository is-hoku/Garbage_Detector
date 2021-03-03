from yolo import YOLO
from yolo import detect_video
from timeit import default_timer as timer
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from cv_bridge import CvBridge
import rospy
import numpy as np
import pyrealsense2 as rs
import threading


class Garbage_Detecter():
    def __init__(self):
        self.garbage_in_can = 0
        self.emergency_stop = 0

    def callback(self, data):
        rospy.loginfo('callback called')
        if data==1:
            rospy.loginfo("Pressed emergency_stop")
            self.emergency_stop = 1
        else:
            self.emergency_stop = 0
            rospy.loginfo("Not pressed emergency_stop")

    def GarbageInCan(self):
        # self.garbage_in_can = rospy.Subscriber("garbage_in_can", Int8, self.callback)
        self.garbage_in_can = 0 # TEST
        # rospy.spin()
    def EmergencyStop(self):
        rospy.loginfo("emergencystop function")
        # self.emergency_stop = 0
        rospy.Subscriber("emergency_stop", Int8, self.callback)
        # if sub==1:
        #     rospy.loginfo("Pressed emergency_stop")
        #     self.emergency_stop = 1
        # else:
        #     self.emergency_stop = 0
        #     rospy.loginfo("Not pressed emergency_stop")
        # self.emergency_stop = rospy.Subscriber("emergency_stop", Int8, self.callback)
        # self.emergency_stop = 0 # TEST
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('bottle_place')
    rospy.loginfo('bottle_place node started')
    garbage = Garbage_Detecter()
    video_path = 0
    # output_path = './output.avi'
    # detect_video(YOLO(), video_path)
    thread1 = threading.Thread(target=garbage.GarbageInCan)
    thread2 = threading.Thread(target=garbage.EmergencyStop)
    thread3 = threading.Thread(target=detect_video, args=(YOLO(),video_path, garbage.garbage_in_can, garbage.emergency_stop))
    thread1.start()
    thread2.start()
    thread3.start()
