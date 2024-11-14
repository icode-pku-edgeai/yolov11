from ultralytics import YOLO
 
if __name__=="__main__":
 
    # pth_path=r"G:\yolov8\ultralytics-main\ultralytics-main\runs\detect\train17\weights\best.pt"
    # Load a model
    #model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO('/code/yolo11/ultralytics-main/ultralytics/models/yolo/detect/runs/detect/yolo11s_allDataset/weights/best.pt')  # load a custom trained model
 
    # Export the model yolo11s_tank_opset13
    model.export(format='onnx',opset=17,simplify=True)