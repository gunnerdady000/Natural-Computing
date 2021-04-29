import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon  # Just used for checking if inside obstacle...
import time

"""
Class to implement the environment simulation.

start_loc - [x,y] value pair for the robot starting location
end_loc - [x,y] value pair for the robot ending location
end_loc - [[x1,y1], [x2,y2], ...] list of points for defining a polygon for the obstacle
"""
class Environment(object):

    # Initialize class
    def __init__(self, start_loc, end_loc, obstacle):
        self.start_loc = start_loc  # Point for the starting location
        self.end_loc = end_loc      # Point for the target location
        self.obstacle = Polygon(obstacle)    # Polygon representing the obstacle

        self.ending_size = 1  # Radius around ending considered "in"
        self.bot_speed = 1    # Number of units bot moves per time step

        self.cur_pos = start_loc
        self.cur_time = 0

        # Define rewards
        self.time_reward = -1/1000  # Reward for spending a unit of time
        self.end_reward = 100         # Reward for finding end
        self.invalid_reward = -10    # Reward for being in an invalid spot
        self.move_closer_reward = 0.01

        # Path taken by the robot
        self.path = np.array([start_loc])

    """
    Display the map.
    """
    def display(self, wait=False):
        plt.clf()

        bounds = self.obstacle.bounds
        y_max = np.max([np.abs(bounds[1]), np.abs(bounds[3])]) * 2
        plt.ylim(-y_max, y_max)

        # Draw the start point (Blue)
        plt.plot(self.start_loc[0], self.start_loc[1], marker='o', color="blue", markersize=15)

        # Draw the end point (green)
        plt.plot(self.end_loc[0], self.end_loc[1], marker='o', color="green", markersize=20)
        # plt.gca().add_patch(plt.Circle(self.end_loc, self.ending_size, color='g'))

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
            plt.pause(0.25)  # brief wait
        else:
            plt.show()
        return True

    """
    Return the current state (position) of the bot in the environment
    """
    def get_state(self):
        return self.cur_pos

    """
    Reset the environment to its initial state
    """
    def reset(self):
        self.path = np.array([self.start_loc])
        self.cur_pos = self.start_loc
        self.cur_time = 0

    """
    Check if position is inside obstacle.
    """
    def in_obstacle(self, pos):
        return self.obstacle.contains(Point(pos))

    """
    Get the simulated time since start.
    """
    def get_time(self):
        return self.cur_time

    """
    Perform an action in the environment and return the results.
    """
    def step(self, drive_dir):
        new_pos = self.move(self.cur_pos, drive_dir)

        # Calculate the rewards and check if done
        reward = self.reward(new_pos)
        done = self.at_end(new_pos)

        # Finish the move
        self.cur_pos = new_pos
        self.path = np.concatenate((self.path, [new_pos]), axis=0)

        self.cur_time += 1

        return self.cur_pos, reward, done

    """
    Move in the direction and return the new position.
    """
    def move(self, pos, dir):
        # Get the new position
        return np.array([pos[0] + self.bot_speed * np.cos(dir),
                        pos[1] + self.bot_speed * np.sin(dir)])

    """
    Return the distance between the two points.
    """
    def dist(self, pt1, pt2):
        return np.sqrt(np.square(pt1[0] - pt2[0]) + np.square(pt1[1] - pt2[1]))

    """
    Return the appropriate reward for the current state.
    """
    def reward(self, pos):
        # We found the end!!!
        if self.at_end(pos):
            return self.end_reward

        # Nope.
        if self.in_obstacle(pos):
            return self.invalid_reward

        # Otherwise, just wasting time. Give small reward for slightly closer
        old_dist = self.dist(self.cur_pos, self.end_loc)
        new_dist = self.dist(pos, self.end_loc)
        dist_reward = (old_dist-new_dist) / self.bot_speed * self.move_closer_reward
        if np.isnan(dist_reward):
            print("Bad reward")
            return self.time_reward
        return self.time_reward + dist_reward

    """
    Check if we have reached the ending state.
    """
    def at_end(self, pos):
        return self.dist(pos, self.end_loc) < self.ending_size


"""
Class to implement replay of past experiences for training.
"""
class Buffer(object):
    # Initialize class
    def __init__(self, capacity=6000, batch_size=25):
        self.capacity = capacity
        self.batch_size = batch_size

        # Index in the buffer for the next item (roll when reach capacity)
        self.index = 0

        # Track if the buffer has been filled with values yet
        self.filled = False

        # Arrays of values from past experiences
        self.states = np.zeros((self.capacity, 2))
        self.actions = np.zeros((self.capacity, 1))
        self.rewards = np.zeros((self.capacity, 1))
        self.new_states = np.zeros((self.capacity, 2))

    """
    Add a new experience to the buffer. Will replace an old one if reached capacity.
    """
    def add(self, state, action, reward, new_state):
        # Store the new results
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.new_states[self.index] = new_state

        # Increment. Roll if needed.
        self.index += 1
        if self.index >= self.capacity:
            self.filled = True
            self.index = 0

    """
    Get a sampling of past experiences from the buffer.
    """
    def get_batch(self):
        # Get number of items in the buffer we can pull from
        if not self.filled:
            buffer_range = self.index
        else:
            buffer_range = self.capacity

        # Get a random list of indices to use
        batch_indices = np.random.choice(buffer_range, self.batch_size)

        # Get the sampling from the buffer
        state_batch = self.states[batch_indices]
        action_batch = self.actions[batch_indices]
        reward_batch = self.rewards[batch_indices]
        new_state_batch = self.new_states[batch_indices]

        return state_batch, action_batch, reward_batch, new_state_batch


"""
Class to generate noise for action selection.
Taken directly from here: https://keras.io/examples/rl/ddpg_pendulum/
"""
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def generate(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

"""
Class to manage the reinforcement learning training. 
Values for training and the neural nets taken from:
https://keras.io/examples/rl/ddpg_pendulum/
"""
class RL(object):
    # Initialize class
    def __init__(self, environment, live_plotting=True):
        self.env = environment
        self.live_plotting = live_plotting

        # Create the neural nets for actor and critic
        self.actor_model = self.make_actor()
        self.critic_model = self.make_critic()

        self.actor_target = self.make_actor()
        self.critic_target = self.make_critic()

        self.actor_target.set_weights(self.actor_model.get_weights())
        self.critic_target.set_weights(self.critic_model.get_weights())

        self.reward_list = []

        # Buffer for storing experiences
        self.buffer = Buffer()

        # Noise generation for actions
        std_dev = 0.25
        self.noise_gen = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1)) # TODO: are these good parameters?

        # Constants that probably need to be adjusted
        self.tau = 0.01  # Rate of target training
        self.gamma = 0.99  # Future rewards
        self.critic_optimizer = tf.keras.optimizers.Adam(0.002)
        #self.actor_optimizer = tf.keras.optimizers.Adam(0.001)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=0.0001)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        self.max_episode_time = 1000


    def make_actor(self):
        last_init = tf.random_uniform_initializer(minval=-np.pi/20, maxval=np.pi/20)

        inputs = layers.Input(shape=(2,))
        out = layers.Dense(32, activation="relu")(inputs)
        out = layers.Dense(32, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Make the bounds for angle
        outputs = outputs * (np.pi)
        actor = tf.keras.Model(inputs, outputs)
        return actor

    def make_critic(self):
        # State as input
        state_input = layers.Input(shape=(2))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(1))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(64, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for given state-action
        critic = tf.keras.Model([state_input, action_input], outputs)

        return critic

    """
    Train for the nets for the given number of episodes
    """
    def train(self, num_episodes):

        for ep in range(num_episodes):
            self.env.reset()
            prev_state = self.env.get_state()
            episode_reward = 0
            done = False

            print("ep"+str(ep)+", total reward: "+str(episode_reward)+", time: "+str(self.env.get_time()), end="")
            while not done and self.env.get_time() < self.max_episode_time:
                # Get the action from the actor model
                action = self.get_action(prev_state, False) #TODO: turned random off

                # Recieve new state and reward from environment.
                state, reward, done = self.env.step(action)

                # Add the new experience to the buffer
                self.buffer.add(prev_state, action, reward, state)
                episode_reward += reward

                # Train based on a batch from the buffer
                state_batch, action_batch, reward_batch, new_state_batch = self.buffer.get_batch()

                self.update_nets(state_batch, action_batch, reward_batch, new_state_batch)

                prev_state = state

                if (self.env.get_time() % 10):
                    print("\rep" + str(ep) + ", total reward: " + str(episode_reward) + ", time: " + str(self.env.get_time()), end="")

            print("\rep" + str(ep) + ", total reward: " + str(episode_reward) + ", time: " + str(self.env.get_time()))
            self.reward_list.append(episode_reward)
            print(episode_reward)
            # Plot every certain number of episodes
            #if self.live_plotting and ep % 10 == 0:
            self.plot_rewards(self.reward_list, False)

        return self.reward_list

    def plot_rewards(self, rewards, wait=True):
        plt.clf()

        plt.plot(range(len(rewards)), rewards, color="blue")
        plt.xlabel("Episode Number")
        plt.ylabel("Total Rewards")

        # Either draw and continue or wait for close figure
        if (not wait):
            plt.draw()
            plt.pause(0.25)  # brief wait
        else:
            plt.show()
        return True

    """
    Train and update the neural nets.
    """
    def update_nets(self, state_batch, action_batch, reward_batch, new_state_batch):
        # Convert batch arrays to tensor for training
        state_batch = tf.convert_to_tensor(state_batch)
        action_batch = tf.convert_to_tensor(action_batch)
        reward_batch = tf.convert_to_tensor(reward_batch)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        new_state_batch = tf.convert_to_tensor(new_state_batch)

        # Train the Critic Model
        with tf.GradientTape() as tape:
            # Reward for the action + best predicted value for the state
            target_action_batch = tf.convert_to_tensor(self.actor_target(new_state_batch, training=True))
            y = reward_batch + self.gamma * self.critic_target([new_state_batch, target_action_batch], training=True)
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Train the Actor Model
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given by the critic for our actions
            actor_loss1 = -tf.math.reduce_mean(critic_value)
            actor_loss = -tf.math.reduce_mean(reward_batch)


        # Compute the weights gradient
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)

        # Apply the gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        # Update the two target networks
        self.update_target_net(self.actor_target, self.actor_model)
        self.update_target_net(self.critic_target, self.critic_model)

    """
    Update the weights on the target neural network
    """
    def update_target_net(self, target_net, model_net):
        # Access the weights directly
        target_weights = target_net.variables
        model_weights = model_net.variables

        # Update the target weights based on the model weights (slowly)
        for (tw, mw) in zip(target_weights, model_weights):
            tw.assign(mw * self.tau + tw * (1 - self.tau))

    """
    Get an action given the current state. Optionally apply noise for exploration.
    """
    def get_action(self, state, noise=False):
        # Get the action from the model
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0) #TODO: wut
        sampled_actions = tf.squeeze(self.actor_model(tf_state)).numpy()

        # Add noise to the action
        if noise:
            sampled_actions = sampled_actions + self.noise_gen.generate()

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, -np.pi * 2, np.pi * 2)

        return np.squeeze(legal_action).item()

"""
Create a map to demonstrate movement and the environment
"""
def demo_map():
    # Create the environment
    obstacle = [[-2, 11],
                [5, 13],
                [5, -16],
                [-6, -14],
                [-8, 4]]
    env = Environment([-20, 0], [20, 0], obstacle)

    env.step(0)
    env.step(np.pi/4)
    env.step(0)
    env.step(np.pi/4)
    env.step(np.pi/4)
    env.step(np.pi/4)
    env.step(0)
    env.display(True)

def main():
    # Create the environment
    obstacle = [[-2, 11],
                [5, 13],
                [5, -16],
                [-6, -14],
                [-8, 4]]
    env = Environment([-20, 0], [20, 0], obstacle)

    # Train the neural nets in the RL
    rl = RL(env)

    rl.train(5)
    rl.plot_rewards(rl.reward_list)

    # Plot the rewards over time

    # Do a run
    env.reset()
    done = False
    while not done and env.get_time() < 1000:
        env.display(False)
        action = rl.get_action(env.get_state())
        state, reward, done = env.step(action)



if __name__ == '__main__':
    main()