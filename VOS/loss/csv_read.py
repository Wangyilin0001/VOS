# python D:\GAN-NET\MUGAN-me\MUGAN\csv_read.py
# Perform graph analysis on the generated each_loss.csv
# python D:\GAN-NET\MUGAN-me\MUGAN\loss\csv_read.py
import csv
import re
import matplotlib.pyplot as plt

# Read the CSV file and extract the numbers
with open(r'E:\MUGAN-me\MUGAN\loss\each_loss_2_1_20.csv', 'r') as file:
	data = file.read().strip()

numbers = [float(x) for x in re.findall(r':(-?\d+(?:\.\d+)?)', data)]


# Store the y value of each color
red_y = []
green_y = []
blue_y = []
orange_y = []


# Create a chart
fig, ax = plt.subplots(figsize=(10, 6))
num = []
# Traverse each string in the array
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

# Connect the points of each color and draw
# ax.plot(range(len(red_y)), red_y, color='red', marker='o', linestyle='-', markersize=1,label='G_GAN')                         #G_GAN
#ax.plot(range(len(green_y)), green_y, color='green', marker='o', linestyle='-',markersize=1, label='G_L1')                  #G_L1
#ax.plot(range(len(blue_y)), blue_y, color='blue', marker='o', linestyle='-', markersize=1,label='D_real')                   #D_real
ax.plot(range(len(orange_y)), orange_y, color='orange', marker='o', linestyle='-', markersize=1,label='D_fake')             #D_fake

# Set the chart properties
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Data Visualization')
ax.legend()

# Show the chart
plt.show()