import numpy as np

number1 = int(input())
number2 = int(input())
number3 = int(input())

array = np.array([number1, number2, number3])
print(array.max())
print(array.argmax())