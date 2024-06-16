# -*- coding: utf-8 -*-

from abcd1234 import f
import numpy as np

# List of test cases: each entry is a tuple (q, r_expected)
test_cases = [
    ((30, 28.583, 11.745, 299.627, 20.978, 0.01), (0.063, 0.098, 0.587)),
    ((0, 0, 0, 0, 0, 0), (0.009, 0.032, 0.654)),
    ((50, 50, 50, 0, 0, 0), (0.385, 0.324, 0.190)),
    # Add more test cases here
]

# Loop through each test case
for i, (q, r_expected) in enumerate(test_cases):

    print(f"Test case {i+1}:")
    print("q:" + str(q))

    # Convert angles from degrees to radians
    q = [np.deg2rad(q[0]), 
         np.deg2rad(q[1]), 
         np.deg2rad(q[2]), 
         np.deg2rad(q[3]), 
         np.deg2rad(q[4]), 
         q[5]]

    # Calculate the result
    r = f(q)

    # Format the results to three decimal places
    r = tuple(f"{value:.3f}" for value in r)
    r_float = tuple(float(value) for value in r)

    # Print the results
    
    print("r expected:   " + str(r_expected))
    print("r calculated: " + str(r_float))

    # print the offset between expected and calculated position
    differences = [r_expected[j] - r_float[j] for j in range(len(r_expected))]
    labels = ['x', 'y', 'z']
    print("Differences:")
    for label, diff in zip(labels, differences):
        print(f"  différence en {label}: {diff:.3f}")
    
    print()

