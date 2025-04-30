import numpy as np


a = np.array([[1, 3, 4],
              [45, 66, 76],
              [0, 9, 4],
              [12, 14, 90],
              [39, 71, 83],
              [27, 20, 5]])
# your code here
row_index = int(input())
column_index = int(input())
print(a[row_index, column_index])