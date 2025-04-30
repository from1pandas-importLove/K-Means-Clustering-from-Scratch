import numpy as np

start  = int(input())
end = int(input())
number_of_intervals = int(input())

array = np.linspace(start, end, number_of_intervals)
print(array[-2])
