"""
HW4_Fractals.py
CSC449
Mason Rud, Micah Runner, Samuel Ryckman
4/30/21
"""
import numpy as np
import matplotlib.pylab as plt
import copy as cp


# prints a line in a specific orientation of length line_len
def print_line(orientation, c_pos):
    """
    0 -> draw pixels to the right
    1 -> draw pixels downward
    2 -> draw pixels upward
    3 -> draw pixels to the left
    """
    i = 0
    # right
    if orientation == 0:
        while i < line_len:
            c_pos[1] += 1
            display_map[c_pos[0]][c_pos[1]] = 100
            i += 1
    # down
    elif orientation == 1:
        while i < line_len:
            c_pos[0] += 1
            display_map[c_pos[0]][c_pos[1]] = 100
            i += 1
    # up
    elif orientation == 2:
        while i < line_len:
            c_pos[0] -= 1
            display_map[c_pos[0]][c_pos[1]] = 100
            i += 1
    # left
    elif orientation == 3:
        while i < line_len:
            c_pos[1] -= 1
            display_map[c_pos[0]][c_pos[1]] = 100
            i += 1


# updates the L string with the rules for a koch curve
def koch_curve(L):
    """
    variables : F
    constants : + −
    start  : F
    rules  : (F → F+F−F−F+F)
    """
    temp_L = ""
    if L == "":
        temp_L = "F"
    else:
        i = 0
        while i < len(L):
            if L[i] == "F":
                temp_L += "F+F−F−F+F"
            else:
                temp_L += L[i]
            i += 1
    return temp_L


# updates the L string with the rules for a dragon curve
def dragon_curve(L):
    """
    variables: F G
    constants: + −
    start: F
    rules: (F → F+G), (G → F-G)
    angle: 90°
    """
    temp_L = ""
    if L == "":
        temp_L = "F"
    else:
        i = 0
        while i < len(L):
            if L[i] == "F":
                temp_L += "F+G"
            elif L[i] == "G":
                temp_L += "F-G"
            else:
                temp_L += L[i]
            i += 1
    return temp_L


# updates the L string with the rules for a binary tree
def generate_tree(L):
    """
    variables: 0 1
    constants: [ ]
    start: 0
    rules: (1 → 11), (0 → 1[0]0)
    """
    temp_L = ""
    if L == "":
        temp_L = "0"
    else:
        i = 0
        while i < len(L):
            if L[i] == "0":
                temp_L += "1[0]0"
            elif L[i] == "1":
                temp_L += "11"
            else:
                temp_L += L[i]
            i += 1
    return temp_L


# prints the koch curve on a matrix
def print_koch_curve(L):
    curr_pos = [0, 0]
    orientation = 0
    i = 0
    while i < len(L):
        # print straight line
        if L[i] == "F":
            print_line(orientation, curr_pos)
        # turn right 90 degrees
        elif L[i] == "+":
            # right
            if orientation == 0:
                orientation = 1
            # down
            elif orientation == 1:
                orientation = 3
            # up
            elif orientation == 2:
                orientation = 0
            # left
            elif orientation == 3:
                orientation = 2
        # turn left 90 degrees
        elif L[i] == "−":
            # right
            if orientation == 0:
                orientation = 2
            # down
            elif orientation == 1:
                orientation = 0
            # up
            elif orientation == 2:
                orientation = 3
            # left
            elif orientation == 3:
                orientation = 1
        i += 1


# prints the dragon curve on a matrix
def print_dragon_curve(L):
    curr_pos = [90, 90]
    orientation = 0
    i = 0
    while i < len(L):
        # print straight line
        if L[i] == "F" or L[i] == "G":
            print_line(orientation, curr_pos)
        # turn right 90 degrees
        elif L[i] == "+":
            # right
            if orientation == 0:
                orientation = 1
            # down
            elif orientation == 1:
                orientation = 3
            # up
            elif orientation == 2:
                orientation = 0
            # left
            elif orientation == 3:
                orientation = 2
        # turn left 90 degrees
        elif L[i] == "-":
            # right
            if orientation == 0:
                orientation = 2
            # down
            elif orientation == 1:
                orientation = 0
            # up
            elif orientation == 2:
                orientation = 3
            # left
            elif orientation == 3:
                orientation = 1
        i += 1


# prints a line in a specific orientation of length c_len
def print_branch(orientation, c_pos, c_len):
    """
    0 -> draw pixels upward
    1 -> draw pixels at a 45 to the right
    2 -> draw pixels at a 45 to the left
    """
    l = 0
    # down_s = 0
    if orientation == 0:
        while l < c_len:
            c_pos[0] += 1
            display_map[c_pos[0]][c_pos[1]] = 100
            l += 1
    # up_s = 1
    elif orientation == 1:
        while l < c_len:
            c_pos[0] -= 1
            display_map[c_pos[0]][c_pos[1]] = 100
            l += 1
    # left_s = 2
    elif orientation == 2:
        while l < c_len:
            c_pos[1] -= 1
            display_map[c_pos[0]][c_pos[1]] = 100
            l += 1
    # right_s = 3
    elif orientation == 3:
        while l < c_len:
            c_pos[1] += 1
            display_map[c_pos[0]][c_pos[1]] = 100
            l += 1
    # down_left_45 = 4
    elif orientation == 4:
        while l < c_len:
            c_pos[0] += 1
            c_pos[1] -= 1
            display_map[c_pos[0]][c_pos[1]] = 100
            l += 1
    # down_right_45 = 5
    elif orientation == 5:
        while l < c_len:
            c_pos[0] += 1
            c_pos[1] += 1
            display_map[c_pos[0]][c_pos[1]] = 100
            l += 1
    # up_right_45 = 6
    elif orientation == 6:
        while l < c_len:
            c_pos[0] -= 1
            c_pos[1] += 1
            display_map[c_pos[0]][c_pos[1]] = 100
            l += 1
    # up_left_45 = 7
    elif orientation == 7:
        while l < c_len:
            c_pos[0] -= 1
            c_pos[1] -= 1
            display_map[c_pos[0]][c_pos[1]] = 100
            l += 1


# prints the binary on a matrix
def print_tree(L):
    curr_pos = [999, 499]
    orientation = 1
    prev_or = []
    prev_pos = []
    i = 0
    while i < len(L):
        # print straight line
        if L[i] == "0" or L[i] == "1":
            print_branch(orientation, curr_pos, line_len)
        # store current position and turn left 45
        elif L[i] == "[":
            prev_pos.append(cp.copy(curr_pos))
            prev_or.append(orientation)
            # update orientation
            # down
            if orientation == 0:
                orientation = 5
            # up
            elif orientation == 1:
                orientation = 7
            # left
            elif orientation == 2:
                orientation = 4
            # right
            elif orientation == 3:
                orientation = 6
            # down left 45
            elif orientation == 4:
                orientation = 0
            # down right 45
            elif orientation == 5:
                orientation = 3
            # up right 45
            elif orientation == 6:
                orientation = 1
            # up left 45
            elif orientation == 7:
                orientation = 2
        # restore current position and turn right 45
        elif L[i] == "]":
            curr_pos = cp.copy(prev_pos.pop())
            orientation = prev_or.pop()
            # update orientation
            # down
            if orientation == 0:
                orientation = 4
            # up
            elif orientation == 1:
                orientation = 6
            # left
            elif orientation == 2:
                orientation = 7
            # right
            elif orientation == 3:
                orientation = 5
            # down left 45
            elif orientation == 4:
                orientation = 2
            # down right 45
            elif orientation == 5:
                orientation = 0
            # up right 45
            elif orientation == 6:
                orientation = 3
            # up left 45
            elif orientation == 7:
                orientation = 1
        i += 1


# updates the L string with the rules for a cantor set
def cantor_set(L):
    """
    variables: A B
    constants: none
    start: A
    rules: (A → ABA), (B → BBB)
    """
    temp_L = ""
    if L == "":
        temp_L = "A"
    else:
        i = 0
        while i < len(L):
            if L[i] == "A":
                temp_L += "ABA"
            elif L[i] == "B":
                temp_L += "BBB"
            else:
                temp_L += L[i]
            i += 1
    return temp_L


# prints the cantor set on a matrix
def print_cantor_set(L, c_row):
    curr_pos = 0
    section_len = cols/len(L)
    i = 0
    while i < len(L):
        # print straight line
        if L[i] == "A":
            k = 0
            while k < section_len:
                display_map[c_row][int(k+i*section_len)] = 100
                k += 1
        # turn right 90 degrees
        elif L[i] == "B":
            k = 0
            while k < section_len:
                display_map[c_row][int(k + i*section_len)] = 50
                k += 1
        i += 1


"""rows = 3
cols = 100
iterations = 3"""
rows = 1700
cols = 3250
iterations = 5
"""rows = 250
cols = 250
iterations = 6"""
line_len = 10
L_str = ""

display_map = np.zeros((rows, cols))
display_map[0][0] = 100
j = 0

# Cantor set
"""while j < iterations:
    L_str = cantor_set(L_str)
    print(L_str)
    print_cantor_set(L_str, j)
    j += 1
plt.title("Cantor Set Fractal, "
          + str(iterations) + " Iterations")"""

# Tree
"""while j < iterations:
    L_str = generate_tree(L_str)
    print(L_str)
    j += 1
print_tree(L_str)
plt.title("L System Tree Fractal, "
          + str(iterations) + " Iterations")"""

# koch curve
while j < iterations+1:
    L_str = koch_curve(L_str)
    print(L_str)
    j += 1
print_koch_curve(L_str)
plt.title("Koch Curve Fractal, "
          + str(iterations) + " Iterations")

# dragon curve
"""while j < iterations+1:
    L_str = dragon_curve(L_str)
    print(L_str)
    j += 1
print_dragon_curve(L_str)
plt.title("Dragon Curve Fractal, "
          + str(iterations) + " Iterations")
"""


plt.imshow(display_map)
plt.show()
