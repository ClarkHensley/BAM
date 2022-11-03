#!/usr/bin/env python3

"""
CSE 4683 Bonus Assignment 2 / Make-Up Quiz 2
Clark Hensley   ch3136

Implement a dynamic BAM NN that would receive a number of pattern pairs, build the BAM network, and memorize the pairs.
"""

import numpy as np

class Association:
    def __init__(self, set_, target):
        # the default np.array objects have an undefined second dimension,
        # so, instead, we reshape into a 2-d array, (1,n), so that we,
        # effectively, have an n-dimensional vector. This makes the
        # numpy linear algebra methods behave better
        self.set = set_.reshape(1, len(set_))
        self.target = target.reshape(1, len(target))
        self.matrix = np.dot(self.set.T, self.target)

def main():
    # Step 0: Define target associations
    
    #### Example Values I found online
    # assoc_1 = Association(np.array([1, 1, -1, -1]), np.array([1, -1]))
    # assoc_2 = Association(np.array([1, -1, 1, -1]), np.array([1, 1]))

    #### Examples from Canvas
    assoc_1 = Association(np.array([1, 1, 1, 1, 1, 1]), np.array([1, 1, 1]))
    assoc_2 = Association(np.array([-1, -1, -1, -1, -1, -1]), np.array([-1, -1, -1]))
    assoc_3 = Association(np.array([1, 1, -1, -1, 1, 1]), np.array([1, -1, 1]))
    assoc_4 = Association(np.array([-1, -1, 1, 1, -1, -1]), np.array([-1, 1, -1]))

    assocs = [assoc_1, assoc_2, assoc_3, assoc_4]

    print("Associations to be stored in the BAM:")
    for a in assocs:
        print(f"\t{a.set} : {a.target}")
    print()

    # Step 1: Create the correlation Matrix
    # np.sum takes a second parameter, the axis to sum across, so we pass in "0"
    # as the second argument
    correlation_matrix = np.sum([a.matrix for a in assocs], 0)

    print("Correlation Matrix generated from the Associations:")
    print(f"{correlation_matrix}")
    print()

    # Step 2: Test that the matrix returns the desired values in the forward direction
    print("Confirming expected values in the forward direction:")
    for a in assocs:
        # sign function to normalize values
        calculated_value = np.sign(np.dot(a.set, correlation_matrix))

        print("\tInput Set:")
        print(f"\t{a.set}")
        print("\tCalculated Value:")
        print(f"\t{calculated_value}")
        print("\tExpected Value:")
        print(f"\t{a.target}")
        print()

    print()
    # Step 3: Test that the matrix returns the desired values in the backward direction
    print("Confirming expected values in the bacward direction:")
    for a in assocs:
        # sign function to normalize values
        calculated_value = np.sign(np.dot(a.target, correlation_matrix.T))

        print("\tInput Set:")
        print(f"\t{a.target}")
        print("\tCalculated Value:")
        print(f"\t{calculated_value}")
        print("\tExpected Value:")
        print(f"\t{a.set}")
        print()

if __name__ == "__main__":
    main()
