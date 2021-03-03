import sys
from tkinter import Tk, ttk
import rospy
from std_msgs.msg import Int8


root = Tk()
root.title(u"Robot Controller")
root.geometry("400x300")
# rospy.init_node("robot_controller")
pub = rospy.Publisher("emergency_stop", Int8)

def stop_button():
    pub.publish(1)
    rospy.loginfo("published 1")

btn = ttk.Button(root, text='緊急停止', command=stop_button)
btn.place(x=150, y=160)

root.mainloop()
