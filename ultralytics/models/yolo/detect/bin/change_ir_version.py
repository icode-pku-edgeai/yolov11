import onnx

def convert_ir_version(model_path, output_path, target_ir_version=8):
    # 加载ONNX模型
    model = onnx.load(model_path)
    
    # 检查当前IR版本
    print(f"Current IR version: {model.ir_version}")
    
    # 修改IR版本
    model.ir_version = target_ir_version
    
    # 保存修改后的模型
    onnx.save(model, output_path)
    print(f"Model saved with IR version {target_ir_version} at {output_path}")

# 使用示例
input_model_path = 'D:\\yolov8_tensorrt\\20231115\\yolov11n_origin.onnx'  # 替换为输入模型路径
output_model_path = 'D:\\yolov8_tensorrt\\20231115\\yolov11n_origin_ir8.onnx'  # 替换为输出模型路径

convert_ir_version(input_model_path, output_model_path)
