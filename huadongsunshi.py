# 创建人 :王壹淋
# 创建时间:2024/10/10;20:26
# 定义一个滑动损失函数
import math

import numpy as np

def huadong_ls(Xl, ls_min=0, ls_max=10, x_min=-5, x_max=5, channel=256):
	# 初始化最小绝对值和对应的ls, x
	min_e_abs = float('inf')
	best_ls = None
	best_x = None
	# 遍历可能的ls和x的取值
	for ls in range(ls_min, ls_max):  # 假设ls的范围
		for x in range(x_min, x_max):  # 假设x的范围
			# for Xl in Xl_range:
			# 计算Y和e
			Y = ls * (ls + Xl) / channel
			e = ls + x - Y

			# 检查当前e的绝对值是否为最小
			if abs(e) < min_e_abs and e != 0:
				min_e_abs = abs(e)
				best_ls = ls
				best_x = x
	# 打印结果
	print(f"最小绝对值的e: {min_e_abs}, 对应的ls: {best_ls}, x: {best_x}")
	return best_ls


l2 = 214
i =0
ls_results = np.empty(0)
ls_results_3 = np.empty(0)
# if __name__ == '__main__':
#
# 	while Xl <= 320:
# 		ls = huadong_ls(Xl, 0, 10, -5, 5, )
# 		ls_results = np.append(ls_results, ls)
# 		# 根据条件更新 results_3
# 		if ls_results.size <= 3:
# 			results_3 = ls_results.copy()  # 如果元素个数少于等于3，则赋值为results
# 		else:
# 			results_3 = ls_results[-3:]  # 如果元素个数大于3，则赋值为results的最新三个值
# 		print(ls_results)
# 		print(results_3)
# 		Xl = ls + Xl
# 		i += 1
# 		print('总共进行了几次计算{}，最后Xl的值为{}'.format(i+1,Xl))
if __name__ == '__main__':

	while l2 <= 320:
		ls = huadong_ls(l2, 0, 10, -5, 5, )
		l2 = ls + l2
		l1 = l2-108
		p1 = 256  / l2
		p2 = 256 / (320 - l1)
		l1_p1 = l1 * p1
		l2_p2 = (l2 - l1) * p2

		ll1 = math.ceil(l1_p1)
		ll2 = int(l2_p2)

		i += 1
		print('总共进行了几次计算{}，l1值为{},变换之后l1_1为{}，向上取整{}，l2的值为{},变换之后l2_2为{}，向下取整{}'.format(i+1,l1,l1_p1,ll1,l2,l2_p2,ll2))