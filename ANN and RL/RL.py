import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon  # Just used for checking if inside obstacle...


class Environment(object):

    # Initialize class
    def __init__(self, start_loc, end_loc, obstacle):
        self.start_loc = start_loc  # Point for the starting location
        self.end_loc = end_loc      # Point for the target location
        self.obstacle = Polygon(obstacle)    # Polygon representing the obstacle

        self.ending_size = 1  # Radius around ending considered "in"
        self.bot_speed = 1    # Number of units bot moves per time step

        self.cur_pos = start_loc

        # Define rewards
        self.time_reward = -1/1000  # Reward for spending a unit of time
        self.end_reward = 1         # Reward for finding end
        self.invalid_reward = -1    # Reward for being in an invalid spot

        # Path taken by the robot
        self.path = np.array([start_loc])

    def display(self, wait=False):
        plt.clf()

        # Draw the start point (Blue)
        plt.plot(self.start_loc[0], self.start_loc[1], marker='o', color="blue")

        # Draw the end point (green)
        plt.gca().add_patch(plt.Circle(self.end_loc, self.ending_size, color='g'))

        # Draw the obstacle (red)
        plt.fill(*self.obstacle.exterior.xy, color="red")

        # Draw the path (yellow)
        plt.plot(*self.path.T, color="yellow")
        last_seg_x = [self.path[-1][0], self.cur_pos[0]]
        last_seg_y = [self.path[-1][1], self.cur_pos[1]]
        plt.plot(last_seg_x, last_seg_y, color="yellow") # Last segment

        # Draw the current position (Black)
        plt.plot(self.cur_pos[0], self.cur_pos[1], marker='o', color="black")

        # Either draw and continue or wait for close figure
        if (not wait):
            plt.draw()
            plt.pause(0.0001)  # brief wait
        else:
            plt.show()
        return True

    def in_obstacle(self, pos):
        return self.obstacle.contains(Point(pos))

    # Perform an action in the environment and return the results
    def step(self, drive_dir):
        new_pos = self.move(self.cur_pos, drive_dir)

        # Calculate the rewards and check if done
        reward = self.reward(new_pos)
        done = self.at_end(new_pos)

        # Finish the move
        self.cur_pos = new_pos
        self.path = np.concatenate((self.path, [new_pos]), axis=0)

        return self.cur_pos, reward, done

    # Move in the direction and return the new position
    def move(self, pos, dir):
        return np.array([pos[0] + self.bot_speed * np.cos(dir),
                        pos[1] + self.bot_speed * np.sin(dir)])

    # Return the distance between the two points
    def dist(self, pt1, pt2):
        return np.sqrt(np.square(pt1[0] - pt2[0]) + np.square(pt1[1] - pt2[1]))

    # Return the appropriate reward for the current state
    def reward(self, pos):
        # We found the end!!!
        if self.at_end(pos):
            return self.end_reward

        # Nope.
        if self.in_obstacle(pos):
            return self.invalid_reward

        # Otherwise, just wasting time
        return self.time_reward

    def at_end(self, pos):
        return self.dist(pos, self.end_loc) < self.ending_size



def main():
    obstacle = [[-10, 20],
                [10, 22],
                [5, -30],
                [-10, -30]]
    env = Environment([-50, 0], [50, 0], obstacle)

    env.display(True)

    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(np.pi / 2)
    env.step(np.pi / 2)
    env.step(np.pi / 2)
    env.step(0)

    env.display(True)


if __name__ == '__main__':
    main()
