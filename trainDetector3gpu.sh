# train
###
 # @Author: your name
 # @Date: 2020-11-20 14:05:26
 # @LastEditTime: 2021-01-30 08:59:36
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edi
 # @FilePath: /darknet/trainDetector3gpu.sh
### 
./darknet detector train data/obj.data cfg/yolo-obj.cfg data/backup/yolo-obj_1002.weights -map -dont_show -mjpeg_port 8090 -gpus 0,1,2


# test
./darknet detector test data/obj.data cfg/yolo-obj.cfg data/backup/yolo-obj_best.weights -dont_show /home/asd/Project/darknet/data/obj/tile_K52E011005_clip_1024_183.png

# test python

python3 darknet_images.py --input /home/asd/Project/darknet/data/obj/ --batch_size 4 --weights data/backup/yolo-obj_best.weights --dont_show --save_labels --data_file /home/asd/Project/darknet/data/obj.data --thresh 0.5


变色立木训练

训练集和验证集划分：
asd@asd-station:~/Project/darknet$ ls  data_attackedtree/obj/*clip_608_[0-1,3-9]*.jpg  > data_attackedtree/train.txt   956个
asd@asd-station:~/Project/darknet$ ls data_attackedtree/obj/*clip_608_2*.jpg > data_attackedtree/valid.txt    156个

训练：
asd@asd-station:~/Project/darknet$ ./darknet detector train data_attackedtree/obj.data cfg/yolo-attackedtree.cfg data/yolov4.conv.137 -map -dont_show -mjpeg_port 8091 -gpus 0,1,2


车辆训练

trian 1383
valid 649

修改cfg文件
网络输入参数
训练参数
yolo头：类别class=2
change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers.


./darknet detector test data_cars/obj.data  cfg/yolo-car.cfg data_cars/backup/yolo-car_best.weights -dont_show data_cars/obj/GOOGLE_mosaic_clip_512_2054.png