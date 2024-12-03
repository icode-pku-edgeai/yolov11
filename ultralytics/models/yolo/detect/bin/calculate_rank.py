# import onnx
# import numpy as np
# import torch
# from torch import linalg

# # 加载 ONNX 模型
# onnx_model = onnx.load('yolo11s.onnx')

# # 获取模型的计算图
# graph = onnx_model.graph

# # 存储卷积层的名称
# conv_layers = []

# # 遍历模型中的所有节点，找到卷积层
# for node in graph.node:
#     if node.op_type == 'Conv':
#         conv_layers.append(node)

# # 如果没有卷积层，则抛出异常
# if len(conv_layers) == 0:
#     raise ValueError("模型中没有找到卷积层！")

# # 遍历所有卷积层
# for conv_layer in conv_layers:
#     # 获取卷积层的权重张量名称（通常是 Conv 的第二个输入）
#     weight_name = conv_layer.input[1]

#     # 查找权重张量
#     weight_tensor = None
#     for initializer in graph.initializer:
#         if initializer.name == weight_name:
#             # 将权重张量转化为 numpy 数组
#             weight_tensor = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(
#                 list(initializer.dims))

#     # 如果找不到权重张量，则跳过
#     if weight_tensor is None:
#         print(f"跳过卷积层 {conv_layer.name}，未找到权重张量")
#         continue

#     # 获取输出通道数 C0（通常为权重张量的第0维）
#     C0 = weight_tensor.shape[0]

#     # 将权重矩阵转换为 PyTorch Tensor
#     weight_tensor = torch.tensor(weight_tensor)

#     # 计算权重矩阵的奇异值
#     # 我们只需要对权重矩阵的每个过滤器（过滤器是权重的前两个维度）计算奇异值
#     U, S, Vh = linalg.svd(weight_tensor.view(C0, -1), full_matrices=False)
    
#     # 最大奇异值 λmax
#     λmax = S[0].item()  # 获取最大奇异值

#     # 计算大于 λmax/2 的奇异值数量 r
#     r = torch.sum(S > (λmax / 2)).item()

#     # 计算标准化的秩：r / C0
#     rank_normalized = r / C0

#     # 输出计算结果
#     # print(f"卷积层 {conv_layer.name} 的最大奇异值 λmax: {λmax}")
#     # print(f"大于 λmax/2 的奇异值数量 r: {r}")
#     # print(f"标准化秩 r / C0: {rank_normalized}\n")
#     print(f"卷积层 {conv_layer.name} 标准化秩 r / C0: {rank_normalized}")



import onnx
import numpy as np
from scipy.linalg import svd

def compute_normalized_rank(weight_matrix):
    """
    计算矩阵的归一化秩，统计大于最大奇异值一半的奇异值个数，并除以输出通道数。
    """
    # 计算矩阵的奇异值
    _, s, _ = svd(weight_matrix, full_matrices=False)
    max_singular_value = s[0]  # 最大奇异值
    threshold = max_singular_value / 2.0
    
    # 统计大于最大奇异值一半的奇异值个数
    count = np.sum(s > threshold)
    
    # 归一化秩
    normalized_rank = count / weight_matrix.shape[0]  # 输出通道数
    return normalized_rank

def extract_convolutional_layers(onnx_model_path):
    """
    从 ONNX 模型中提取卷积层的相关信息，并计算归一化秩。
    """
    # 加载 ONNX 模型
    model = onnx.load(onnx_model_path)
    
    # 遍历模型的每一层
    for node in model.graph.node:
        if node.op_type == 'Conv':
            # 获取卷积层的输入和输出
            input_name = node.input[0]
            output_name = node.output[0]
            
            # 获取卷积权重
            weight_initializer = None
            for init in model.graph.initializer:
                if init.name == node.input[1]:  # 第二个输入是卷积权重
                    weight_initializer = init
                    break
            
            if weight_initializer is None:
                print(f"卷积层 {node.name} 没有找到权重初始化器")
                continue
            
            # 提取权重矩阵的形状
            weight_array = np.frombuffer(weight_initializer.raw_data, dtype=np.float32)
            weight_shape = weight_initializer.dims
            assert len(weight_shape) == 4, f"卷积权重矩阵的形状应该是4D，但得到了{len(weight_shape)}D"
            
            # 重塑权重矩阵为 (输出通道数, 输入通道数 * 卷积核大小的平方)
            output_channels, input_channels, kernel_height, kernel_width = weight_shape
            reshaped_weight_matrix = weight_array.reshape(output_channels, input_channels * kernel_height * kernel_width)
            
            # 计算归一化秩
            normalized_rank = compute_normalized_rank(reshaped_weight_matrix)
            
            # 打印卷积层的相关信息
            print(f"卷积层: {node.name},归一化秩: {normalized_rank:.4f}")
            # print(f"  输入通道数: {input_channels}")
            # print(f"  输出通道数: {output_channels}")
            # print(f"  卷积核大小: {kernel_height}x{kernel_width}")
            # print(f"  归一化秩: {normalized_rank:.4f}")
            # print("-" * 40)

# 示例：从 ONNX 文件中提取卷积层信息并计算归一化秩
onnx_model_path = 'yolo11s.onnx'
extract_convolutional_layers(onnx_model_path)
