from ultralytics import YOLO
# # yolo=YOLO("D:\\yolov8\\ultralytics\\models\\yolo\\detect\\yolov8n.pt",task="detect")
# yolo=YOLOv10("/code/yolov10/ultralytics/models/yolov10/runs/detect/train_yolov10s_allDatasets/weights/best.pt",task="detect")
yolo=YOLO("/code/yolo11/ultralytics-main/ultralytics/models/yolo/detect/runs/detect/yolo11s_allDataset/weights/best.pt",task="detect")
#/home/lizhehan/
# result=yolo(source="D:\\yolov8\\ultralytics\\assets")#检测图像
# result=yolo(source="/datasets/datasets/new_dataset/rename/images/val_tank+tk",save=False)#检测图像
result=yolo(source="/datasets/datasets/new_dataset/rename/images/val_all_datasets/000239.jpg",save=True)#检测图像
# result=yolo(source="/datasets/datasets/object_detect/origin_video/",save=True)#检测图像
# # result=yolo(source="D:\\yolov8_tensorrt\\100.avi")#检测视频
# result=yolo(source="D:\\yolov8_tensorrt\\1.mp4")#检测视频
# # result=yolo(source="screen")#检测桌面
# # result=yolo(source=0)#检测摄像头
# #