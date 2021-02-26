# Garbage_Detector
## :information_source: Information
### Realsenseをusbで直接使う(yolo_camera.py)   
Node name: ```bottle_place```   
Publisher: ```bottle_point```, ```bottle_or_person```   
Subscriber: ```garbage_in_can``` 
### RealsenseのtopicをSubscribeして使う(allprocess.py)   
Node name: ```bottle_place```   
Publisher: ```bottle_point```, ```bottle_or_person```   
Subscriber: ```garbage_in_can```, ```/camera/color/image_raw```, ```/camera/depth/image_rect_raw```   
***caution***   
It's not completely!   

## :notebook: Note
- Kerasをインストールして，keras-yolo3とlabelImgをディレクトリに持ってくる．   
- keras-yolo3の中に```yolo.py```, ```yolo_video.py```, ```yolo_camera.py```, ```realsensecv.py```を置く．   
- ```roslaunch realsense2_camera rs_camera.launch```を実行してRealSenseをPublishする
