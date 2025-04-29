# python D:\GAN-NET\MUGAN-me\MUGAN\csv_read.py
# 将生成的each_loss.csv进行画图分析
# python D:\GAN-NET\MUGAN-me\MUGAN\loss\csv_read.py
import csv
import re
import matplotlib.pyplot as plt

# 读取 CSV 文件并提取数字
with open(r'E:\MUGAN-me\MUGAN\loss\each_loss_2_1_20.csv', 'r') as file:
	data = file.read().strip()

numbers = [float(x) for x in re.findall(r':(-?\d+(?:\.\d+)?)', data)]


# 存储每个颜色的 y 值
red_y = []
green_y = []
blue_y = []
orange_y = []


# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))
num = []
# 遍历数组中的每一个字符串
for i, row_data in enumerate(numbers):
	num.append(row_data)
	#print(num)
	if len(num) == 6:
		x = [(num[0] - 1) * 1000 + num[1]]
		y = [num[2], num[3], num[4], num[5]]
		red_y.append(y[0])
		green_y.append(y[1])
		blue_y.append(y[2])
		orange_y.append(y[3])
		num = []

# 连接每种颜色的点并绘制
# ax.plot(range(len(red_y)), red_y, color='red', marker='o', linestyle='-', markersize=1,label='G_GAN')                         #G_GAN
#ax.plot(range(len(green_y)), green_y, color='green', marker='o', linestyle='-',markersize=1, label='G_L1')                  #G_L1
#ax.plot(range(len(blue_y)), blue_y, color='blue', marker='o', linestyle='-', markersize=1,label='D_real')                   #D_real
ax.plot(range(len(orange_y)), orange_y, color='orange', marker='o', linestyle='-', markersize=1,label='D_fake')             #D_fake

# 设置图表属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Data Visualization')
ax.legend()

# 显示图表
plt.show()