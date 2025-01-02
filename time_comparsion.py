import pandas as pd

# 定义一个函数来计算 LU 分解的理论时间
def compute_theoretical_time(n, tfps):
    # 计算 FLOPs：2/3 * n^3
    flops = (2 / 3) * (n ** 3)
    # 将 TFLOPS 转换为 FLOP/s
    flops_per_second = tfps * 10**12
    # 理论时间 (秒)
    theoretical_time_sec = flops / flops_per_second
    # 转换为毫秒
    return theoretical_time_sec * 1000

# 读取 CSV 文件
df = pd.read_csv('lu_factorization_times.csv')

# 假设你使用的 GPU 性能是 170 TFLOPS（NVIDIA RTX 4090D）
gpu_tfops = 170  # 单精度浮点运算性能 (TFLOPS)

# 提取矩阵大小的整数部分
df['Matrix Size'] = df['Matrix Size'].apply(lambda x: int(x.split('x')[0]))

# 添加理论时间和性能差距的列
df['Theoretical Time (ms)'] = df['Matrix Size'].apply(lambda x: compute_theoretical_time(x, gpu_tfops))
df['Performance Difference'] = df['Theoretical Time (ms)'] / df['Time (ms)'] * 100

# 输出结果
print(df)

# 可选：保存到新 CSV 文件
df.to_csv('lu_factorization_comparison_4090.csv', index=False)