import numpy as np
import matplotlib.pylab as plt


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

rows = 1700
cols = 3250
iterations = 6
line_len = 4
L_str = ""
display_map = np.zeros((rows, cols))
display_map[0][0] = 100
j = 0
# konch curve
while j < iterations + 1:
    L_str = koch_curve(L_str)
    print(L_str)
    j += 1
print_koch_curve(L_str)
plt.title("Koch Curve Fractal, "
          + str(iterations) + " Iterations")

"""
rows = 250
cols = 250
iterations = 10
line_len = 4
L_str = ""
display_map = np.zeros((rows, cols))
display_map[0][0] = 100
j = 0

# dragon curve
while j < iterations+1:
    L_str = dragon_curve(L_str)
    print(L_str)
    j += 1
print_dragon_curve(L_str)
plt.title("Dragon Curve Fractal, "
          + str(iterations) + " Iterations")

"""
plt.imshow(display_map)
plt.show()


