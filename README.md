# 训练相关基本操作
## 环境
+ python
+ torch
+ yolov11是基于yolov8开发的，已经在ultralytics库中替代了v8了
+ 官方更新的ultralytics并不自带yolov10
## 数据集
+ 数据格式，和yolov5相同
```
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val

```
## 命令行执行，和yolov8一模一样
+ 详细内容参见default.yaml
```
yolo task=detect mode=train model=yolov8x.yaml data=mydata.yaml epochs=1000 batch=16
```
+ task:目标检测detect、分割segment、分类classify等等
+ mode：训练train、验证val、预测predict
+ model:模型配置yaml文件或者加载pt权重文件
+ pretrained:或者可以设置model为yaml文件，然后pretrained为pt文件进行自适应的部分迁移学习
+ data：数据集yaml
+ epochs:迭代次数
+ batch：视显存大小而定
+ imgsz：图片尺度
+ device：gpu设备
+ optimizer：优化器，默认sgd,可选adam等等
+ source:想要推理的目录，可以是图片、视频、文件夹、屏幕、摄像头
+ patience：早停机制
+ workers：0肯定可以，其他数值请自行尝试
+ resume：断点存续
+ iou:iou阈值
+ conf:置信度阈值
+ half：fp16推理
+ max_det：最大检测数
+ format:导出格式，默认torchscript，可选onnx、engine等
+ dynamic:动态导出
+ simplify：简化
+ opset：onnx版本
## 代码执行，和yolov8一模一样
+ 训练
```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # 从YAML中构建一个新模型
model = YOLO('yolov8n.pt')  #加载预训练的模型(推荐用于训练)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从YAML构建并传递权重

# Train the model
model.train(data='coco128.yaml', epochs=100, imgsz=640)


```
+ 验证
```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  #加载官方模型
model = YOLO('path/to/best.pt')  # 加载自己训练的模型

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

```
+ 推理
```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

# 目标检测后处理
boxes = results[0].boxes
boxes.xyxy  # box with xyxy format, (N, 4)
boxes.xywh  # box with xywh format, (N, 4)
boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
boxes.xywhn  # box with xywh format but normalized, (N, 4)
boxes.conf  # confidence score, (N, 1)
boxes.cls  # cls, (N, 1)
boxes.data  # raw bboxes tensor, (N, 6) or boxes.boxes .

# 实例分割后处理
masks = results[0].masks  # Masks object
masks.segments  # bounding coordinates of masks, List[segment] * N
masks.data  # raw masks tensor, (N, H, W) or masks.masks 

# 目标分类后处理
results = model(inputs)
results[0].probs  # cls prob, (num_class, )


```

+ 导出
```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('path/to/best.pt')  # load a custom trained

# Export the model
model.export(format='onnx')

```
+ 跟踪
```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official detection model
model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
model = YOLO('path/to/best.pt')  # load a custom model

# Track with the model
results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True) 
results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True, tracker="bytetrack.yaml") 


```
+ 基准
```
from ultralytics.yolo.utils.benchmarks import benchmark

# Benchmark
benchmark(model='yolov8n.pt', imgsz=640, half=False, device=0)

```
# 代码基础介绍，和yolov8基本一致
## docker各类硬件的docker file
## docs 文档
## examples 各种推理框架案例
## logs 提供了yolov10的训练日志
## tests 各种测试代码
## ultralytics核心代码
### cfg存放yaml文件
+ datasets各种数据集yaml
+ models各类目标检测模型v3-v10以及rt-detr
+ trackers跟踪类算法botsort和bytetrack
+ default.yaml超参表
### data数据相关代码，数据增强，数据加载等等
### engine模型相关代码
### hub 模型托管平台
### models各类模型调用
#### fastsam和sam:segment anything model
#### nas:neural architecture search
#### rtdetr和utils工具
#### yolo
+ classify分类
+ detect检测
+ obb有向边界框
+ pose姿态检测
+ segment分割
+ world词汇对象检测
+ model.py 父类模型，调用上述子类对象
#### yolov10 v10的所有调用都在这个文件夹
+ model.py 父类模型，调用上述子类对象
+ predict.py检测代码
+ train.py训练代码
+ val.py验证代码
### nn神经网络
#### modules模块
+ block.py子模块，例如C1、C2、C3、C2F、ELAN、RepVGG、bottleneck等等
+ conv.py各种卷积，例如conv、dwconv、ghostconv、cbam、concat等等
+ head.py各种检测头，对应models里面的分类、检测、分割、姿态检测、有向边界框、rtdetr、v10检测头，核心代码在v10Detect中
+ transformer.py各种transformer类的模块
+ utils.py工具
#### autobackend.py推理时动态后端选择
#### tasks.py 
+ 从模型yaml文件中解析组成模型
+ 所有新增的module都需要import，并在parse_model函数中适时调用
### solutions 附属功能的解决方案
### trackers 跟踪实现的代码
+ bot_sort和byte_tracker实现，详细参考文件夹内readme
### utils各类工具
+ autobatch.py:自动batch工具
+ benchmarks.py:多平台对比工具
+ checks.py:检测工具
+ dist.py downloads.py:下载工具
+ errors.py:报错工具
+ files.py:文件工具
+ instance.py：实例对象
+ loss.py：损失函数
+ metrics.py:评价指标
+ ops.py运营工具，例如nms、xyxy2xywh之类的
+ patches.py补丁工具
+ plotting.py绘图工具
+ tal.py 任务对齐学习Task Alignment Learning
+ torch_utils.py torch工具
+ triton.py triton推理工具
+ tuner.py 超参调优工具
# 核心改进点
+ 从yolov5的C3到yolov8的C2f，再到yolov11的C3k2，本质上就是C2f，只是将深层的C2f的kernel定为3
+ C2PSA,从yolov10的PSA借鉴得来，加入了C2f到PSA中
+ EIOU损失函数
+ 加快了训练速度，亲测有效
+ 新增了边缘设备一键部署的支持