import numpy as np


a = np.array([[[10, 11, 12], [13, 14, 15], [16, 17, 18]],
              [[20, 21, 22], [23, 24, 25], [26, 27, 28]],
              [[30, 31, 32], [33, 34, 35], [36, 37, 38]],
              [[40, 41, 42], [43, 44, 45], [46, 47, 48]],
              [[50, 51, 52], [53, 54, 55], [56, 57, 58]],
              [[60, 61, 62], [63, 64, 65], [66, 67, 68]],
              [[70, 71, 62], [73, 74, 65], [76, 77, 78]],
              [[80, 81, 62], [83, 84, 85], [86, 87, 88]]])
# your code here
sub_array_step_index = int(input())
row_step_index = int(input())
element_index = int(input())
print(a[::sub_array_step_index, ::row_step_index, element_index])

