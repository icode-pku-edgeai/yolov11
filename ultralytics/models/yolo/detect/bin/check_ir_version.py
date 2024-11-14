import onnx  
import os  
  
# 加载ONNX模型  
model_path = 'C:\\Users\\li\\Desktop\\repo\\detection\\yolo\\yolov11\\ultralytics\\models\\yolo\\detect\\bin'  # 替换为你的ONNX模型文件路径  
onnx_files = [f for f in os.listdir(model_path) if f.endswith('.onnx')]
for onnx_file in onnx_files:
    file_path = os.path.join(model_path, onnx_file)
    try:
        print(f"---------------------")
        onnx_model = onnx.load(file_path)  
        print(f"model name: {onnx_file}")  # 注意：这里实际上应该使用onnx_model.producer_name和onnx_model.producer_version，但ir_version不是直接可用的属性，这里只是为了说明而故意写错，下面会纠正。  

        # 打印ONNX模型的IR版本相关信息（通过元数据）  
        # 注意：ONNX模型不直接暴露一个叫做“IR version”的字段  
        # 但我们可以通过producer_version和opset_imports来推断一些信息  

        # 打印生成者版本（producer_version），这通常表示生成该ONNX模型的工具或库的版本  
        print(f"Producer Version: {onnx_model.ir_version}")  # 注意：这里实际上应该使用onnx_model.producer_name和onnx_model.producer_version，但ir_version不是直接可用的属性，这里只是为了说明而故意写错，下面会纠正。  
        # 正确的打印方式应该是：  
        print(f"Producer Name: {onnx_model.producer_name}")  
        print(f"Producer Version: {onnx_model.producer_version}")  

        # 打印模型使用的opset版本信息  
        opset_versions = [opset.version for opset in onnx_model.opset_import]  
        print(f"Opset Versions: {opset_versions}")  

        # 注意：ONNX模型的“IR version”通常不是直接暴露的，而是通过opset版本来间接表示模型的兼容性。  
        # 在ONNX中，每个opset版本都引入了一组新的运算符或运算符的更改。  
        # 因此，通过查看模型使用的opset版本，你可以推断出模型与哪些ONNX版本的运算符集兼容。  

        # 纠正上面的错误打印  
        # onnx_model.ir_version 实际上是不存在的属性，这里应该使用onnx.IR_VERSION来获取ONNX库支持的IR版本，但这并不表示模型的IR版本。  
        # 模型的“IR版本”通常与其使用的opset版本和ONNX规范的演变相关，而不是一个直接可查询的属性。  
        # 因此，下面的代码只是展示了如何获取ONNX库自身的IR版本支持信息，而不是模型的IR版本。  
        print(f"ONNX Library IR Version Support: {onnx.IR_VERSION}")
        print(f"---------------------")
    except Exception as e:
        print(f"Failed to load model {onnx_file}: {e}")
