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


if __name__ == '__main__':
    video_path = 0
    output_path = './output.avi'
    detect_video(YOLO(), video_path)
