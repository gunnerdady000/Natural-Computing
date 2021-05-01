"""
ComplexTreeLsys.py
CSC449
Mason Rud, Micah Runner, Samuel Ryckman
4/30/21
"""
import turtle
import copy as cp


def createLSystem(numIters, axiom):
    startString = axiom
    endString = ""
    for i in range(numIters):
        endString = update_string(startString)
        startString = endString

    return endString


def update_string(oldStr):
    newstr = ""
    for ch in oldStr:
        newstr = newstr + generate_tree(ch)

    return newstr


def generate_tree(ch):
    newstr = ""
    if ch == 'F':
        newstr = 'B'   # Rule 1
    elif ch == 'B':
        newstr = 'F[-B]F[+B][B]'   # Rule 2
    else:
        newstr = ch    # no rules apply so keep the character

    return newstr


def draw_tree(aTurtle, instructions, angle, distance):
    prev_pos = []
    pos = []
    for cmd in instructions:
        angle -= 0.05
        if cmd == 'F' or cmd == 'B':
            aTurtle.forward(distance)
        elif cmd == '+':
            aTurtle.left(angle)
        elif cmd == '-':
            aTurtle.right(angle)
        elif cmd == '[':
            prev_pos.append([cp.copy(aTurtle.xcor()), cp.copy(aTurtle.ycor()), aTurtle.heading()])
        elif cmd == ']':
            aTurtle.penup()
            pos = prev_pos.pop()
            aTurtle.goto(pos[0], pos[1])
            aTurtle.setheading(pos[2])
            aTurtle.pendown()


inst = createLSystem(5, "F")   # create the string
print(inst)
t = turtle.Turtle()            # create the turtle
wn = turtle.Screen()

t.up()
t.back(200)
t.down()
t.speed(9999999)
draw_tree(t, inst, 60.0, 50)   # draw the picture
                                 
wn.exitonclick()

