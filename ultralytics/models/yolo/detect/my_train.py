

from ultralytics import YOLO

# model_yaml_path = "/code/yolov10/ultralytics/models/yolov10/my_yolov5m.yaml"
model_yaml_path = "C:\\Users\\li\\Desktop\\repo\\detection\\yolo\\yolov11\\ultralytics\\cfg\models\\11\\yolo11s.yaml"
#数据集配置文件
data_yaml_path = 'tank.yaml'
#预训练模型
pre_model_name = 'yolo11s.pt'

if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    # model = YOLOv10(model_yaml_path)
    #训练模型
    results = model.train(data=data_yaml_path,
                          epochs=500,
                          batch=1,
                          imgsz=640,
                          workers=0,
                          name='test',
                          patience=30)


# from ultralytics.cfg import entrypoint
# arg="yolo detect train model=/code/yolov10/ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml data=/code/yolov10/ultralytics/models/yolov10/my_dataset.yaml batch=64 workers=0"

# entrypoint(arg)