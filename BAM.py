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
    print("Bidirectional Associative Memory Generator:")
    # The association vectors must be the same size, generate the lhs and rhs sizes
    lh_size = generate_size("left-hand side of the Associations")
    rh_size = generate_size("right-hand side of the Associations")
    # Ensure at least one entry
    print()
    print("Values for the Associations may be entered separated by spaces, commas, or commas and white space: (a,b,c) or (a b c) or (a, b, c)")
    print("Values for the BAM must be bits. Please enter 1 and -1 as the values. If you enter 0, these 0s will be converted to -1s. Other values are not valid entries.")
    print()
    assocs = list()
    assocs.append(create_association(lh_size, rh_size))
    # Allow the user to enter an arbitrary number of Associations
    while True:
        print()
        if confirm("Are there more associations to add?"):
            assocs.append(create_association(lh_size, rh_size))
        else:
            break

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
    print("Confirming expected values in the backward direction:")
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


def generate_size(target):
    while True:
        print(f"Please enter the size in bits of the {target}")
        try:
            size = int(input("> "))
            if size < 1:
                raise ValueError
        except ValueError:
            print("Error: The sizes of the associations must be integers greater than 0")
        else:
            if confirm(f"You entered {size}.", "yes"):
                return size
    

def confirm(message, default=None):
    check_str = "Is this correct (y/n)? "
    yes_vals = ("yes", "y")
    no_vals = ("no", "n")
    if isinstance(default, str):
        if default.lower() in yes_vals:
            yes_vals = ("yes", "y", "")
            check_str = "Is this correct (Y/n)? "
        elif default.lower() in no_vals:
            no_vals = ("no", "n", "")
            check_str = "Is this correct (y/N)? "

    while True:
        print(message)
        user_input = input(check_str)
        if user_input.lower() in yes_vals:
            return True
        elif user_input.lower() in no_vals:
            return False
        else:
            print("Error: invalid input")


def create_association(lh_size, rh_size):
    set_ = get_list(lh_size)
    target = get_list(rh_size)
    return Association(np.array(set_), np.array(target))


def get_list(size):
    while True:
        print(f"Please enter {size} values for left-hand side of the association:")
        side = input("> ")
        # The formatting requires that we account for white-space, comma, and whitespace-comma separated strings
        side = side.strip()
        if "," in side:
            side = side.split(",")
        else:
            side = side.split(" ")
        formatted_side = list()
        valid_input = True
        for s in side:
            s = s.strip()
            if not s or s == ",":
                continue
            try:
                formatted_s = int(s)
                if formatted_s == 0:
                    formatted_s = -1
                if formatted_s not in (1, -1):
                    raise ValueError
            except ValueError:
                print("Error: You must enter a list of whitespace, comma, or whitespace-comma separated integers in [1, 0, -1], formatted as (a,b,c) or (a b c) or (a, b, c)")
                valid_input = False
                break
            else:
                formatted_side.append(formatted_s)

        # Invalid input, error already handled, just continue the loop
        if not valid_input:
            continue
        
        elif len(formatted_side) != size:
            print(f"Error: You must enter {size} values for this list")

        else:
            if confirm(f"You entered {formatted_side}", "yes"):
                return formatted_side


if __name__ == "__main__":
    main()
