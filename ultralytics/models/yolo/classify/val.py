# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class ClassificationValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a classification model.
    分类的验证基类
    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.models.yolo.classify import ClassificationValidator

        args = dict(model="yolov8n-cls.pt", data="imagenet10")
        validator = ClassificationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "classify"
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        """Returns a formatted string summarizing classification metrics."""#分类尺度总结字符串格式
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model):
        """Initialize confusion matrix, class names, and top-1 and top-5 accuracy."""#初始化尺度：混淆矩阵、类名、top1-top5精度
        self.names = model.names
        self.nc = len(model.names)
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf, task="classify")
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""#前处理:图片和类别gpu-half/float
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):#尺度更新
        """Updates running metrics with model predictions and batch targets."""
        n5 = min(len(self.names), 5)#top5添加记录
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch["cls"].type(torch.int32).cpu())

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes metrics of the model such as confusion_matrix and speed."""
        #在训练或验证结束时计算和汇总模型的最终指标，如混淆矩阵和速度
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def get_stats(self):
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        #返回一个字典，包含通过处理目标（targets）和预测（predictions）获得的指标
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        """Creates and returns a ClassificationDataset instance using given image path and preprocessing parameters."""
        #分类数据集建立
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for classification tasks with given parameters."""
        #分类任务的dataloader
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """Prints evaluation metrics for YOLO object detection model."""#输出结果
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""#绘制验证样图
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""#绘制带box框的验证样图
        plot_images(
            batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=torch.argmax(preds, dim=1),
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
