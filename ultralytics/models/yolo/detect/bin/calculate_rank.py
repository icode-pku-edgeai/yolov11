import onnx
import numpy as np
import torch
from torch import linalg

# 加载 ONNX 模型
onnx_model = onnx.load('yolo11s_tank_opset13.onnx')

# 获取模型的计算图
graph = onnx_model.graph

# 存储卷积层的名称
conv_layers = []

# 遍历模型中的所有节点，找到卷积层
for node in graph.node:
    if node.op_type == 'Conv':
        conv_layers.append(node)

# 如果没有卷积层，则抛出异常
if len(conv_layers) == 0:
    raise ValueError("模型中没有找到卷积层！")

# 遍历所有卷积层
for conv_layer in conv_layers:
    # 获取卷积层的权重张量名称（通常是 Conv 的第二个输入）
    weight_name = conv_layer.input[1]

    # 查找权重张量
    weight_tensor = None
    for initializer in graph.initializer:
        if initializer.name == weight_name:
            # 将权重张量转化为 numpy 数组
            weight_tensor = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(
                list(initializer.dims))

    # 如果找不到权重张量，则跳过
    if weight_tensor is None:
        print(f"跳过卷积层 {conv_layer.name}，未找到权重张量")
        continue

    # 获取输出通道数 C0（通常为权重张量的第0维）
    C0 = weight_tensor.shape[0]

    # 将权重矩阵转换为 PyTorch Tensor
    weight_tensor = torch.tensor(weight_tensor)

    # 计算权重矩阵的奇异值
    # 我们只需要对权重矩阵的每个过滤器（过滤器是权重的前两个维度）计算奇异值
    U, S, Vh = linalg.svd(weight_tensor.view(C0, -1), full_matrices=False)
    
    # 最大奇异值 λmax
    λmax = S[0].item()  # 获取最大奇异值

    # 计算大于 λmax/2 的奇异值数量 r
    r = torch.sum(S > (λmax / 2)).item()

    # 计算标准化的秩：r / C0
    rank_normalized = r / C0

    # 输出计算结果
    # print(f"卷积层 {conv_layer.name} 的最大奇异值 λmax: {λmax}")
    # print(f"大于 λmax/2 的奇异值数量 r: {r}")
    # print(f"标准化秩 r / C0: {rank_normalized}\n")
    print(f"卷积层 {conv_layer.name} 标准化秩 r / C0: {rank_normalized}")
