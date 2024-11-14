import copy
import onnxruntime as rt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import torch 
import os  
from PIL import Image  
import json
# 前处理
def resize_image(image, size, letterbox_image):
    """
        对输入图像进行resize
    Args:
        size:目标尺寸
        letterbox_image: bool 是否进行letterbox变换
    Returns:指定尺寸的图像
    """
    ih, iw, _ = image.shape#读取原图尺寸hwc
    # print(ih, iw)
    h, w = size#拿到目标尺寸
    # letterbox_image = False
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # print(image.shape)
        # 生成画布，128像素的默认图
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        # 将image放在画布中心区域-letterbox
        image_back[(h-nh)//2: (h-nh)//2 + nh, (w-nw)//2:(w-nw)//2+nw , :] = image
    else:
        image_back = image
        # cv2.imshow("img", image_back)
        # cv2.waitKey()
    return image_back  


def img2input(img):
    img = np.transpose(img, (2, 0, 1))#hwc->chw
    img = img/255#norm,转fp64
    return np.expand_dims(img, axis=0).astype(np.float32)#增加一个维度并转fp32

def std_output(pred):
    """
    将（1，84，8400）处理成（8400， 85）  85= box:4  conf:1 cls:80
    """
    pred = np.squeeze(pred)#压缩掉大小为1的维度
    pred = np.transpose(pred, (1, 0))#两个维度互换
    pred_class = pred[..., 4:]#跳过xywh，单独拿出来类别维度
    pred_conf = np.max(pred_class, axis=-1)#指定轴计算数组的最大值
    pred = np.insert(pred, 4, pred_conf, axis=-1)#将计算出来的最大类别插入到第5个位置，形成xywh+conf的输出
    return pred

def xywh2xyxy(*box):
    """
    将xywh转换为左上角点和左下角点
    Args:
        box:
    Returns: x1y1x2y2
    """
    ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
          box[0] + box[2] // 2, box[1] + box[3] // 2]
    return ret

def get_inter(box1, box2):
    """
    计算相交部分面积
    Args:
        box1: 第一个框
        box2: 第二个狂
    Returns: 相交部分的面积
    """
    x1, y1, x2, y2 = xywh2xyxy(*box1)
    x3, y3, x4, y4 = xywh2xyxy(*box2)
    # 验证是否存在交集
    if x1 >= x4 or x2 <= x3:
        return 0
    if y1 >= y4 or y2 <= y3:
        return 0
    # 将x1,x2,x3,x4排序，因为已经验证了两个框相交，所以x3-x2就是交集的宽
    x_list = sorted([x1, x2, x3, x4])
    x_inter = x_list[2] - x_list[1]
    # 将y1,y2,y3,y4排序，因为已经验证了两个框相交，所以y3-y2就是交集的宽
    y_list = sorted([y1, y2, y3, y4])
    y_inter = y_list[2] - y_list[1]
    # 计算交集的面积
    inter = x_inter * y_inter
    return inter

def get_iou(box1, box2):
    """
    计算交并比： (A n B)/(A + B - A n B)
    Args:
        box1: 第一个框
        box2: 第二个框
    Returns:  # 返回交并比的值
    """
    box1_area = box1[2] * box1[3]  # 计算第一个框的面积
    box2_area = box2[2] * box2[3]  # 计算第二个框的面积
    inter_area = get_inter(box1, box2)
    union = box1_area + box2_area - inter_area   #(A n B)/(A + B - A n B)
    iou = inter_area / union
    return iou
def nms(pred, conf_thres, iou_thres):
    """
    非极大值抑制nms
    Args:
        pred: 模型输出特征图
        conf_thres: 置信度阈值
        iou_thres: iou阈值
    Returns: 输出后的结果
    """
    box = pred[pred[..., 4] > conf_thres]  # 将上一步计算的概率最大的类别进行一次置信度筛选
    cls_conf = box[..., 5:]#拿到剩余的置信度值
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))#将剩余置信度分别对应的类别索引记录
    total_cls = list(set(cls))  # 记录图像内共出现几种物体
    output_box = []
    # 每个预测类别分开考虑
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        temp = box[:, :6]
        for j in range(len(cls)):
            # 记录[x,y,w,h,conf(最大类别概率),class]值
            if cls[j] == clss:
                temp[j][5] = clss
                cls_box.append(temp[j][:6])
        #  cls_box 里面是[x,y,w,h,conf(最大类别概率),class]
        cls_box = np.array(cls_box)
        sort_cls_box = sorted(cls_box, key=lambda x: -x[4])  # 将cls_box按置信度从大到小排序
        # box_conf_sort = np.argsort(-box_conf)
        # 得到置信度最大的预测框
        max_conf_box = sort_cls_box[0]
        output_box.append(max_conf_box)#保留最大的那个
        sort_cls_box = np.delete(sort_cls_box, 0, 0)#删掉最大的那个
        # 对除max_conf_box外其他的框进行非极大值抑制
        while len(sort_cls_box) > 0:
            # 得到当前最大的框
            max_conf_box = output_box[-1]
            del_index = []
            for j in range(len(sort_cls_box)):
                current_box = sort_cls_box[j]
                iou = get_iou(max_conf_box, current_box)
                if iou > iou_thres:
                    # 筛选出与当前最大框Iou大于阈值的框的索引
                    del_index.append(j)
            # 删除这些索引
            sort_cls_box = np.delete(sort_cls_box, del_index, 0)
            if len(sort_cls_box) > 0:
                # 我认为这里需要将clas_box先按置信度排序， 才能每次取第一个
                output_box.append(sort_cls_box[0])
                sort_cls_box = np.delete(sort_cls_box, 0, 0)
    return output_box

def cod_trf(result, pre, after):
    """
    因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上
    Args:
        result:  [x,y,w,h,conf(最大类别概率),class]
        pre:    原尺寸图像
        after:  经过letterbox处理后的图像
    Returns: 坐标变换后的结果,
    """
    res = np.array(result)
    x, y, w, h, conf, cls = res.transpose((1, 0))#转置
    x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)  # 左上角点和右下角的点
    h_pre, w_pre, _ = pre.shape
    h_after, w_after, _ = after.shape
    scale = max(w_pre/w_after, h_pre/h_after)  # 缩放比例
    h_pre, w_pre = h_pre/scale, w_pre/scale  # 计算原图在等比例缩放后的尺寸
    x_move, y_move = abs(w_pre-w_after)//2, abs(h_pre-h_after)//2  # 计算平移的量
    ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
    ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
    ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
    return ret

def draw(res, image, cls):
    """
    将预测框绘制在image上
    Args:
        res: 预测框数据
        image: 原图
        cls: 类别列表，类似["apple", "banana", "people"]  可以自己设计或者通过数据集的yaml文件获取
    Returns:
    """
    for r in res:
        # 画框
        image = cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 0, 0), 1)
        # 表明类别
        text = "{}:{}".format(cls[int(r[5])], \
                               round(float(r[4]), 2))
        h, w = int(r[3]) - int(r[1]), int(r[2]) - int(r[0])  # 计算预测框的长宽
        font_size = min(h/640, w/640) * 3  # 计算字体大小（随框大小调整）
        image = cv2.putText(image, text, (max(10, int(r[0])), max(20, int(r[1]))), cv2.FONT_HERSHEY_COMPLEX, max(font_size, 0.3), (0, 0, 255), 1)   # max()为了确保字体不过界
    # cv2.imshow("result", image)
    # cv2.waitKey()
    return image
def save_txt(result,save_txt_path,ih, iw,annotations):
    with open(save_txt_path, 'w') as f:  
        for boxes in result:  
            x1, y1, x2, y2, conf, cls = boxes  
            # 计算中心点坐标  
            x_center = float((x1 + x2) / 2)  
            y_center = float((y1 + y2) / 2)  
            # 计算宽度和高度  
            width = float(x2 - x1)  
            height =float(y2 - y1)   
            # 写入文件，格式为 cls x y w h，每个值之间用空格分隔  
            f.write(f"{int(cls)} {x_center/iw} {y_center/ih} {width/iw} {height/ih}\n") 
            new_annotation = {  
                "bbox": [x_center-width/2,y_center-height/2,width,height],  
                "category_id": int(cls),  
                "image_id": filename,  
                "score": float(conf)  
            } 
            annotations.append(new_annotation)

    
def calculate_map_by_json(annotations_path,results_file):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Run COCO mAP evaluation
    # Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(annotation_file=annotations_path)
    cocoDt = cocoGt.loadRes(results_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


# 加载配置文件

if __name__ == '__main__':
    config_file = "C:\\Users\\li\\Desktop\\repo\\detection\\yolo\\yolov11\\ultralytics\\models\\yolo\\detect\\tank.yaml"
    input_path = "D:\\datasets\\homemade4\\images\\"  # 输入图片的根目录路径
    out_images_path = "D:\\datasets\\homemade4\\result\\"
    out_labels_path = "D:\\datasets\\homemade4\\labels\\"
    #cuda版本限制了onnxruntime版本，进而限制了IR version
    onnx_path='C:\\Users\\li\\Desktop\\repo\\detection\\yolo\\yolov11\\ultralytics\\models\\yolo\\detect\\bin\\yolo11s_tank_opset13.onnx'
    annotations_path = "D:\\datasets\\tank_debug\\val.json"
    with open(config_file, "r") as config:
        config = yaml.safe_load(config)
    std_h, std_w = 640, 640  # 指定输入尺寸
    dic = config["names"]  # 得到的是模型类别字典
    class_list = list(dic.values())

    #onnx加载
    providers = ['CUDAExecutionProvider']
    sess = rt.InferenceSession(onnx_path,providers=providers)  # yolov8模型onnx格式
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    annotations = [] 
    for filename in os.listdir(input_path):  
        # 检查文件扩展名，只加载图片文件（例如：.jpg, .png, .jpeg, .bmp, .gif）  
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
            img_path = os.path.join(input_path, filename)  
            print(img_path)
            img = cv2.imread(img_path)
            if img.size == 0:
                print("路径有误！")
            # 前处理 letterbox
            img_after = resize_image(img, (std_w, std_h), True)  # （640， 640， 3）
            # 将图像处理成输入的格式
            data = img2input(img_after)#bchw、fp32
            #输入模型onnx
            pred = sess.run([label_name], {input_name: data})[0]  # 输出(8400x84, 84=80cls+4reg, 8400=3种尺度的特征图叠加), 这里的预测框的回归参数是xywh， 而不是中心点到框边界的距离
            #后处理v10
            if pred.shape==(1,300,6):
                result = pred[0, pred[0, :, 4] >= 0.5]
                x1 = result[:, 0]
                y1 = result[:, 1]
                x2 = result[:, 2]
                y2 = result[:, 3]
                # 计算 w 和 h
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                center_x=(x1+x2)/2
                center_y=(y1+y2)/2
                # 创建新的输出数组
                result = np.column_stack((center_x, center_y, w, h, result[:, 4], result[:, 5]))
            else:#后处理v8、v11
                pred = std_output(pred)
                # 置信度过滤+nms
                result = nms(pred, 0.5, 0.4)  # [x,y,w,h,conf(最大类别概率),class]
                # 坐标变换
            if len(result):
                result = cod_trf(result, img, img_after)
                image = draw(result, img, class_list)
                # cv2.imwrite(out_images_path + filename, image)
            else:
            # 保存输出图像   
                pass         
                # cv2.imwrite(out_images_path + filename, img)
            filename = filename.rsplit('.', 1)[0] 
            ih, iw, _ = img.shape
            
            save_txt(result,out_labels_path+filename+'.txt',ih, iw,annotations)     
            
            # save_json(result,out_labels_path+'val.json',filename)
            # cv2.destroyWindow("result") 
    # print(annotations)
    ##输出json
    # with open(out_labels_path+'val.json', 'w') as json_file:  
    #     json.dump(annotations, json_file, indent=4, ensure_ascii=False) 
    ##计算map值
    # calculate_map_by_json(annotations_path,out_labels_path+'val.json')