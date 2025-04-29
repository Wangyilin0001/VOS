# 创建人 :王壹淋
# 创建时间:2024/10/21;16:17
import numpy as np

# 创建一个空的数组
results = np.empty(0)

# 进行200次计算
for i in range(200):
	# 假设每次计算的结果为当前索引的平方（可以根据实际需要修改计算逻辑）
	result = i ** 2

	# 更新数组，将结果添加到数组中
	results = np.append(results, result)

	# 打印当前的结果和数组内容
	print(f"Iteration {i + 1}: Result = {result}, Results Array = {results}")

# 最终的结果数组
print("Final Results Array:", results)