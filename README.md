# Garbage_Detector
## :information_source: Information
### Realsenseをusbで直接使う(yolo_camera.py)   
Node name: bottle_place   
| Publisher        | type                | description                                | Subscriber                   | type              | description                              | 
| ---------------- | ------------------- | ------------------------------------------ | ---------------------------- | ----------------- | ---------------------------------------- | 
| real_coordinate  | geometry_msgs/Point | RealSense中心の座標(カメラ座標系)          | garbage_in_can               | Int8              | ゴミがゴミ箱に捨てられた時 1，それ以外 0 | 
| bottle_or_person | Int8                | 追跡しているのがbottleの時 0，personの時 1 |       | |                        | 
| tracking         | Int8                | 物体認識中は0，追跡中は1                   |  |  |                      | 
| yolo_frame       | sensor_msgs/Image   | 物体認識中の映像                           |                              |                   |                                          | 
| tracking_frame   | sensor_msgs/Image   | 追跡中の映像                               |                              |                   |                                          | 

### RealsenseのtopicをSubscribeして使う(allprocess.py)   
Node name: bottle_place    
| Publisher        | type                | description                                | Subscriber                   | type              | description                              | 
| ---------------- | ------------------- | ------------------------------------------ | ---------------------------- | ----------------- | ---------------------------------------- | 
| real_coordinate  | geometry_msgs/Point | RealSense中心の座標(カメラ座標系)          | garbage_in_can               | Int8              | ゴミがゴミ箱に捨てられた時 1，それ以外 0 | 
| bottle_or_person | Int8                | 追跡しているのがbottleの時 0，personの時 1 | /camera/color/image_raw      | sensor_msgs/Image | RealSenseのRGB画像                       | 
| tracking         | Int8                | 物体認識中は0，追跡中は1                   | /camera/depth/image_rect_raw | sensor_msgs/Image | RealSenseのDepth画像                     | 
| yolo_frame       | sensor_msgs/Image   | 物体認識中の映像                           |                              |                   |                                          | 
| tracking_frame   | sensor_msgs/Image   | 追跡中の映像                               |                              |                   |                                          | 

***caution***   
It's not completely!   

### 環境変数   
ROS_IP=192.168.10.3   
ROS_IP(another)=192.168.10.4   
ROS_MASTER_URI=http://192.168.10.3:11311   

## :notebook: Note
- Kerasをインストールして，keras-yolo3とlabelImgをディレクトリに持ってくる．   
- keras-yolo3の中に```yolo.py```, ```yolo_video.py```, ```yolo_camera.py```, ```realsensecv.py```を置く．   
- ```roslaunch realsense2_camera rs_camera.launch```を実行してRealSenseをPublishする
