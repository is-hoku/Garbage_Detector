# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import cv2
import time
import colorsys
import os
from timeit import default_timer as timer
import sys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
# from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

from realsensecv import RealsenseCapture

#packages for ROS Publisher
import rospy
# from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Int8
from geometry_msgs.msg import Point
# from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def start_node():
    from sensor_msgs.msg import Image
    # rospy.init_node('bottle_place')
    rospy.loginfo('bottle_place node started')
    pub = rospy.Publisher("real_coordinate", Point)
    pub_flag = rospy.Publisher("bottle_or_person", Int8)
    pub_track = rospy.Publisher("tracking", Int8)
    pub_frame1 = rospy.Publisher("yolo_frame", Image)
    pub_frame2 = rospy.Publisher("tracking_frame", Image)
    return pub, pub_flag, pub_track, pub_frame1, pub_frame2

# def call_back(ros_data):
#     np_arr = np.fromstring(ros_data.data, np.uint8)


class YOLO(object):
    from PIL import Image, ImageFont, ImageDraw
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        # print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, pub):
        from PIL import Image, ImageFont, ImageDraw
        start = timer()
        bottle = False
        person = False
        ro,lo,bo,to,ro2,lo2,bo2,to2 = 0,0,0,0,0,0,0,0

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                # K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        human_list = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))
            if (predicted_class=="bottle") & (score >= 0.25):
                bottle = True
                ro = right
                lo = left
                bo = bottom
                to = top
            #place of human who is holding a bottle
            elif (predicted_class=="person") & (bottle) & (score >=0.6):
                person = True
                human_list.append([right, left, bottom, top])


            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        # print("human_list = ", human_list)
        if person:
            near = 640000
            for i in human_list:
                distance = (((i[0]-i[1])//2+i[1])-((ro-lo)//2+lo))**2+(((i[2]-i[3])//2+i[3])-((bo-to)//2+to))**2
                if near>distance:
                    near = distance
                    ro2, lo2, bo2, to2 = i[0], i[1], i[2], i[3]
        # print("Tracked Person = ", ro2, lo2, bo2, to2)

        end = timer()
        print("一回の検出にかかる時間", end - start)
        return image, bottle, person, ro, lo, bo, to, ro2, lo2, bo2, to2

    def close_session(self):
        self.sess.close()

# def detect_video(yolo, frames, video_path, output_path=""):
def detect_video(yolo, video_path, garbage_in_can, emergency_stop):
    from PIL import Image, ImageFont, ImageDraw
    #Start ROS node
    pub, pub_flag, pub_track, pub_frame1, pub_frame2 = start_node()
    vid = RealsenseCapture()
    vid.start()
    bridge = CvBridge()

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    worldy = 0.0

    while True:
        pub_track.publish(0)
        ret, frames, _ = vid.read()
        frame = frames[0]
        depth_frame = frames[1]
        image = Image.fromarray(frame)
        image, bottle, person, right, left, bottom, top, right2, left2, bottom2, top2 = yolo.detect_image(image, pub)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.imshow("result", result)
        yolo_frame = bridge.cv2_to_imgmsg(result, "bgr8")
        # yolo_frame = result
        pub_frame1.publish(yolo_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if (bottle==False) or (person==False):
            continue



    # ------------------------------Tracking-----------------------------------
        # tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        # tracker_type = tracker_types[7]
        tracker = cv2.TrackerCSRT_create()
        tracker2 = cv2.TrackerCSRT_create()

        # setup initial location of window
        left, right, top, bottom = left, right, top, bottom
        r,h,ci,w = top, bottom-top, left, right-left  # simply hardcoded the values r, h, c, w
        frame_b, frame_g, frame_r = frame[:,:,0], frame[:,:,1], frame[:,:,2]
        hist_b = cv2.calcHist([frame_b[top:bottom, left:right]],[0],None,[256],[0,256])
        hist_g = cv2.calcHist([frame_g[top:bottom, left:right]],[0],None,[256],[0,256])
        hist_r = cv2.calcHist([frame_r[top:bottom, left:right]],[0],None,[256],[0,256])
        cv2.normalize(hist_b, hist_b,0,255,cv2.NORM_MINMAX)
        cv2.normalize(hist_g, hist_g,0,255,cv2.NORM_MINMAX)
        cv2.normalize(hist_r, hist_r,0,255,cv2.NORM_MINMAX)
        track_window = (ci, r, w, h)

        r2,h2,ci2,w2 = top2, bottom2-top2, left2, right2-left2  # simply hardcoded the values r, h, c, w
        hist_bp = cv2.calcHist([frame_b[top2:bottom2, left2:right2]],[0],None,[256],[0,256])
        hist_gp = cv2.calcHist([frame_g[top2:bottom2, left2:right2]],[0],None,[256],[0,256])
        hist_rp = cv2.calcHist([frame_r[top2:bottom2, left2:right2]],[0],None,[256],[0,256])
        cv2.normalize(hist_bp, hist_bp,0,255,cv2.NORM_MINMAX)
        cv2.normalize(hist_gp, hist_gp,0,255,cv2.NORM_MINMAX)
        cv2.normalize(hist_rp, hist_rp,0,255,cv2.NORM_MINMAX)
        track_window2 = (ci2, r2, w2, h2)
        # print(left2, top2, right2-left2, bottom2-top2)
        cv2.imwrite('bottledetect.jpg', frame[r:r+h, ci:ci+w])
        cv2.imwrite('persondetect.jpg', frame[r2:r2+h2, ci2:ci2+w2])

        # set up the ROI for tracking
        roi = frame[r:r+h, ci:ci+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        ok = tracker.init(frame, track_window)
        ok2 = tracker2.init(frame, track_window2)

        track_thing = 0 #bottle
        pts = Point()
        pts2 = Point()
        untrack = 0

        while(1):
            ret ,frames, depth = vid.read()
            frame = frames[0]
            depth_frame = frames[1]

            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

                # apply meanshift to get the new location
                ok, track_window = tracker.update(frame)
                x,y,w,h = track_window

                ok, track_window2 = tracker2.update(frame)
                x2,y2,w2,h2 = track_window2

                # Draw it on image
                img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                if not track_thing:
                    img2 = cv2.rectangle(img2, (x2,y2), (x2+w2,y2+h2), 255,2)
                else:
                    img2 = cv2.rectangle(img2, (x2, y2), (x2+w2, y2+h2),(0, 0, 255), 2)
                cv2.imshow('Tracking',img2)
                tracking_frame = bridge.cv2_to_imgmsg(img2, "bgr8")
                # tracking_frame = img2
                pub_frame2.publish(tracking_frame)

                # https://www.intelrealsense.com/wp-content/uploads/2020/06/Intel-RealSense-D400-Series-Datasheet-June-2020.pdf
                total, cnt = 0, 0
                for i in range(3):
                    for j in range(3):
                        dep = depth.get_distance(np.maximum(0, np.minimum(i+x+w//2, 639)), np.maximum(0, np.minimum(j+y+h//2, 479)))
                        if (dep)!=0:
                            total += dep
                            cnt += 1
                            print('dep = ', dep)
                if cnt!=0:
                    worldz = total/cnt
                    # 人にぶつからないように距離を確保するため
                    if worldz<1.0:
                        worldz=0
                else:
                    worldz = 0

                total2, cnt2 = 0, 0
                for i in range(3):
                    for j in range(3):
                        dep2 = depth.get_distance(np.maximum(0, np.minimum(i+x2+w2//2, 639)), np.maximum(0, np.minimum(j+y2+h2//2, 479)))
                        if dep2!=0:
                            total2 += dep2
                            cnt2 += 1
                if cnt2!=0:
                    worldz2 = total2/cnt2
                    if worldz2<1.0:
                        worldz2=0
                else:
                    worldz2 = 0

                # print('worldz', worldz)
                # print('worldz2', worldz2)
                if (worldz == 0) or (worldz2 == 0):
                    # break
                    worldx, worldy = 0, 0
                    worldx = 0
                    pts.x, pts.y, pts.z = 0.0, 0.0, 0.0
                    worldx2, worldy2 = 0, 0
                    pts2.x, pts2.y, pts2.z = 0.0, 0.0, 0.0
                else:
                    # focus length = 1.93mm, distance between depth cameras = about 5cm, a pixel size = 3um
                    if (track_thing==0):
                        #bottle Tracking
                        u_ud = (0.05*1.88*10**(-3))/(3*10**(-6)*worldz)
                        # print('u_ud', u_ud)
                        # print('x, y =', (x+w//2)-(img2.shape[1]//2), (img2.shape[0]//2)-(y+h//2))
                        # 深度カメラとカラーカメラの物理的な距離を考慮した項(-0.3*u_ud)
                        # これらの座標は物体を見たときの左の深度カメラを基準とする
                        worldx = 0.05*(x+w//2 - (img2.shape[1]//2) - 0.3*u_ud)/u_ud
                        worldy = 0.05*((img2.shape[0]//2) - (y+h))/u_ud
                        print('x,y,z = ', worldx, worldy, worldz-1.0)
                        pts.y, pts.z, pts.x = float(worldx), float(worldy), float(worldz)-1.0

                    else:
                        #human Tracking
                        u_ud = (0.05*1.88*10**(-3))/(3*10**(-6)*worldz2)
                        worldx2 = 0.05*(x2+w2//2 - (img2.shape[1]//2) - 0.3*u_ud)/u_ud
                        worldy2 = 0.05*((img2.shape[0]//2) - (y2+h2))/u_ud
                        print('x2,y2,z2 = ', worldx2, worldy2, worldz2-1.0)
                        pts2.x, pts2.y, pts.z = float(worldx2), float(worldy2), float(worldz2)-1.0

                print("track_thing = ", track_thing)

                frame_b, frame_g, frame_r = frame[:,:,0], frame[:,:,1], frame[:,:,2]
                hist_b2 = cv2.calcHist([frame_b[y: y+h, x: x+w]],[0],None,[256],[0,256])
                hist_g2 = cv2.calcHist([frame_g[y: y+h, x: x+w]],[0],None,[256],[0,256])
                hist_r2 = cv2.calcHist([frame_r[y: y+h, x: x+w]],[0],None,[256],[0,256])
                cv2.normalize(hist_b2, hist_b2,0,255,cv2.NORM_MINMAX)
                cv2.normalize(hist_g2, hist_g2,0,255,cv2.NORM_MINMAX)
                cv2.normalize(hist_r2, hist_r2,0,255,cv2.NORM_MINMAX)
                hist_bp2 = cv2.calcHist([frame_b[y2: y2+h2, x2: x2+w2]],[0],None,[256],[0,256])
                hist_gp2 = cv2.calcHist([frame_g[y2: y2+h2, x2: x2+w2]],[0],None,[256],[0,256])
                hist_rp2 = cv2.calcHist([frame_r[y2: y2+h2, x2: x2+w2]],[0],None,[256],[0,256])
                cv2.normalize(hist_bp2, hist_bp2,0,255,cv2.NORM_MINMAX)
                cv2.normalize(hist_gp2, hist_gp2,0,255,cv2.NORM_MINMAX)
                cv2.normalize(hist_rp2, hist_rp2,0,255,cv2.NORM_MINMAX)
                comp_b = cv2.compareHist(hist_b, hist_b2, cv2.HISTCMP_CORREL)
                comp_g = cv2.compareHist(hist_g, hist_g2, cv2.HISTCMP_CORREL)
                comp_r = cv2.compareHist(hist_r, hist_r2, cv2.HISTCMP_CORREL)
                comp_bp = cv2.compareHist(hist_bp, hist_bp2, cv2.HISTCMP_CORREL)
                comp_gp = cv2.compareHist(hist_gp, hist_gp2, cv2.HISTCMP_CORREL)
                comp_rp = cv2.compareHist(hist_rp, hist_rp2, cv2.HISTCMP_CORREL)
                # print('compareHist(b)', comp_b)
                # print('compareHist(g)', comp_g)
                # print('compareHist(r)', comp_r)
                # print('compareHist(bp)', comp_bp)
                # print('compareHist(gp)', comp_gp)
                # print('compareHist(rp)', comp_rp)
                # print("garbage_in_can", garbage_in_can)
                # 追跡を止める条件は，bottle追跡中にヒストグラムが大きく変化するか枠が無くなるまたはpersonを見失う，もしくはperson追跡中にヒストグラムが大きく変化するか枠が無くなるまたはゴミがゴミ箱に入れられた，
                if ((track_thing==0 and ((comp_b<=0.1)or(comp_g<=0.1)or(comp_r<=0.1) or track_window==(0, 0, 0, 0))) or (track_window2==(0, 0, 0, 0))
                or (track_thing==1 and ((comp_bp<=0.)or(comp_gp<=0.)or(comp_rp<=0.)))):
                    untrack += 1
                    print("untrack = ", untrack)
                    if untrack>=30:
                        print("追跡が外れた！\n")
                        break
                elif (track_thing==0 and (x+w>640 or x<0) and (y+h>480 or y<0)) or (track_thing==1 and (x2+w2>640 or x2<0) and (y2+h2>480 or y2<0)):
                    untrack+=1
                    print("untrack = ", untrack)
                    if untrack>=50:
                        print("枠が画面外で固まった")
                        break
                elif (track_thing==1 and garbage_in_can==1):
                    print("ゴミを捨てたため追跡を終えます")
                    break
                else:
                    untrack = 0

                # ポイ捨ての基準はbottleを追跡していて，地面から10cmのところまで落ちたか，bottleを見失ったかつ見失う前のフレームでの高さがカメラの10cmより下
                print('track_window = ', track_window)
                if (((worldy<=-0.10) or (track_window==(0,0,0,0) and (worldy<0.5))) and (not track_thing)):
                    print("ポイ捨てした！\n")
                    track_thing = 1 #human

                if track_thing==0:
                    tracking_point = pts
                    if not (pts.x==0.0 and pts.y==0.0 and pts.z==0.0):
                        pub.publish(tracking_point)
                    flag = 0 #bottle
                else:
                    tracking_point = pts2
                    if not (pts2.x==0.0 and pts2.y==0.0 and pts2.z==0.0):
                        pub.publish(tracking_point)
                    flag = 1 #person

                pub_flag.publish(flag)


                k = cv2.waitKey(1) & 0xff
                if (k == 27) or emergency_stop: # dev
                # if emergency_stop: # ops
                    print("program is stoped!")
                    sys.exit(0)
            else:
                break
            pub_track.publish(1)


    yolo.close_session()
