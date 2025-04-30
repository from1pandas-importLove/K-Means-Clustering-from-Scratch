import numpy as np

number_of_rows_and_columns = int(input())
number_to_fullfill_array = int(input())

if number_to_fullfill_array == 1:
    print(np.ones((number_of_rows_and_columns, number_of_rows_and_columns)))
else:
    print(np.zeros((number_of_rows_and_columns, number_of_rows_and_columns)))
