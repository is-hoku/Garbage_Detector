# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import cv2
import time
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

#packages for ROS Publisher
import rospy
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge


def start_node():
    rospy.init_node('bottle_location')
    rospy.loginfo('bottle_location node started')
    pub = rospy.Publisher("bottle_points", Int32MultiArray)
    return pub


class YOLO(object):
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

        print('{} model, anchors, and classes loaded.'.format(model_path))

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
        # start = timer()
        # bottle = [-1, -1]
        bottle = False
        ro,lo,bo,to = 0,0,0,0

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

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
            print(label, (left, top), (right, bottom))
            if (predicted_class=="bottle") & (score >= 0.7):
                # bottle = [right-left, bottom-top]
                bottle = True
                ro = right
                lo = left
                bo = bottom
                to = top







                # # setup initial location of window
                # r,h,ci,w = top, bottom-top, left, right-left  # simply hardcoded the values r, h, c, w
                # track_window = (left, top, right-left, bottom-top) # x, y, w, h / c, r, w, h
                #
                # # set up the ROI for tracking
                # roi = image[r:r+h, ci:ci+w]
                # hsv_roi =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                # roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                # cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                #
                # # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
                # term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                #
                # while(1):
                #     # ret ,frame = cap.read()
                #
                #     if ret == True:
                #         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                #         dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                #
                #         # apply meanshift to get the new location
                #         ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                #
                #         # Draw it on image
                #         pts = cv2.boxPoints(ret)
                #         pts = np.int0(pts)
                #         cv2.polylines(image,[pts],True, 255,2)
                #         # cv2.imshow('img2',img2)
                #         print(pts)
                #         bottle = Int32MultiArray(data=pts)
                #         pub.publish(bottle)
                #
                #         k = cv2.waitKey(60) & 0xff
                #         if k == 27:
                #             break
                #         # else:
                #             # cv2.imwrite(chr(k)+".jpg",img2)
                #     else:
                #         break















            # bottle = Int32MultiArray(data=bottle)
            # pub.publish(bottle)

            #あとでコメントアウト外す
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.

            #あとでコメントアウト外す
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        # end = timer()
        # print(end - start)
        return image, bottle, ro, lo, bo, to

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    # import cv2
    #Start ROS node
    pub = start_node()

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    # vid.set(cv2.CAP_PROP_FPS, 10)
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        ret, frame = vid.read()
        image = Image.fromarray(frame)
        # print(image.size)
        image, bottle, right, left, bottom, top = yolo.detect_image(image, pub)



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
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



        if bottle==False:
            continue



# ------------------------------MeanSift------------------------------------

        # setup initial location of window
        left, right, top, bottom = left, right, top, bottom
        r,h,ci,w = top, bottom-top, left, right-left  # simply hardcoded the values r, h, c, w
        track_window = (left, top, right-left, bottom-top) # x, y, w, h / c, r, w, h
        print(left, top, right-left, bottom-top)

        # set up the ROI for tracking
        roi = frame[r:r+h, ci:ci+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        start = timer()
        while(1):
            ret ,frame = vid.read()
            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

                # apply meanshift to get the new location
                ret, track_window = cv2.meanShift(dst, track_window, term_crit)
                x,y,w,h = track_window
                x,y,w,h = x, y, w, h

                # Draw it on image
                # pts = cv2.boxPoints(ret)
                # pts = np.int0(pts)
                # img2 = cv2.polylines(frame,[pts],True, 255,2)
                img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                cv2.imshow('img2',img2)
                pts = [x, y, x+w, y+h]
                print(pts)
                if (x<=5 or x+w>=635)or(y<=5 or y+h>=475):
                    print("追跡が外れた！\n")
                    break
                # cv2.waitKey(0)
                # time.sleep(3)
                bottle = Int32MultiArray(data=pts)
                pub.publish(bottle)


                k = cv2.waitKey(60) & 0xff
                if k == 27:
                    break
                # else:
                #     cv2.imwrite(chr(k)+".jpg",img2)
            else:
                break

            end = timer()
            print(end - start)
            if (end-start)>=15:
                break























        # ret, frame = vid.read()
        # image = Image.fromarray(frame)
        # image, bottle = yolo.detect_image(image, pub, ret, frame)
        # if !bottle:
        #     continue

        # result = np.asarray(image)
        # curr_time = timer()
        # exec_time = curr_time - prev_time
        # prev_time = curr_time
        # accum_time = accum_time + exec_time
        # curr_fps = curr_fps + 1
        # if accum_time > 1:
        #     accum_time = accum_time - 1
        #     fps = "FPS: " + str(curr_fps)
        #     curr_fps = 0
        # cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.50, color=(255, 0, 0), thickness=2)
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        #
        # if isOutput:
        #     out.write(result)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    yolo.close_session()
