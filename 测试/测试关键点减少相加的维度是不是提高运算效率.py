import torch
import time

# 初始化参数
tensor_size = (3, 256, 256)  # 例如，3通道（RGB），每个通道大小为256x256
batch_size = 1000

# 初始化张量
a = torch.rand((batch_size, *tensor_size), device='cuda')
b = torch.rand((batch_size, *tensor_size), device='cuda')
zero_tensor = torch.zeros((batch_size, *tensor_size), device='cuda')

# 创建一个大部分为零的张量，其中有一些非零值
sparse_tensor = torch.zeros((batch_size, *tensor_size), device='cuda')
sparse_indices = torch.randperm(batch_size * tensor_size[1] * tensor_size[2])[:22]  # 选择22个随机位置
sparse_values = torch.rand((22, tensor_size[0]), device='cuda')
for idx, val in zip(sparse_indices, sparse_values):
    batch_idx = idx // (tensor_size[1] * tensor_size[2])
    pixel_idx = idx % (tensor_size[1] * tensor_size[2])
    x = pixel_idx // tensor_size[2]
    y = pixel_idx % tensor_size[2]
    sparse_tensor[batch_idx, :, x, y] = val

# 测试大批量的有效数字相加
repetitions = 1000  # 增加重复次数
start_time = time.time()
for _ in range(repetitions):
    result_1 = a + b
torch.cuda.synchronize()  # 等待所有操作完成
end_time = time.time()
print(f"Time for a + b: {(end_time - start_time) / repetitions:.10f} seconds")

# 测试大批量的数值与0相加
start_time = time.time()
for _ in range(repetitions):
    result_2 = a + zero_tensor
torch.cuda.synchronize()  # 等待所有操作完成
end_time = time.time()
print(f"Time for a + 0: {(end_time - start_time) / repetitions:.10f} seconds")

# 测试大批量的数值与非常少数的非零值相加
start_time = time.time()
for _ in range(repetitions):
    result_3 = a + sparse_tensor
torch.cuda.synchronize()  # 等待所有操作完成
end_time = time.time()
print(f"Time for a + sparse_tensor: {(end_time - start_time) / repetitions:.10f} seconds")
