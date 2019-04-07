import numpy as np


def percent_round_int(percent, x):
    return np.round(percent * x).astype(int)


class MyBar():
    """
        Wrap the parameters and the dynamics related
        to the bar the agent interacts with.
    """

    def __init__(self, width, height, grid_width, grid_height):
        """
        Set the parameters of the bar (size) and its initial position

        Arguments:
        ----------
        - `width`: Bar width
        - `height`: Bar height
        - `grid_width`: Grid width
        - `grid_height`: Grid height

        """
        self.width = width
        self.height = height

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.vel = 0.0

        self.size = (width, height)
        self.center = (
            grid_width / 2 - self.width / 2,
            grid_height - height - 3)

    def reset(self):
        """
        Resets to initial state of the bar.
        Center position, null speed.
        """
        self.center = (
            self.grid_width / 2 - self.width / 2,
            self.grid_height - self.height - 3)
        self.vel = 0.0

    def update(self, dx):
        """
            Dynamics of the bar.

            Arguments:
            ----------
            - `dx`: Real-valued force
                    (towards left when <0, right otherwise)
        """

        # Update velocity and position
        self.vel += dx
        self.vel *= 0.9

        x, y = self.center
        n_x = x + self.vel

        """
            Keeps the bar inside the grid.
            No bounce, null speed if grid limits
            are reached.
        """
        if n_x - self.width / 2.0 <= 0:
            self.vel = 0.0
            n_x = self.width / 2.0

        if n_x + self.width / 2.0 >= self.grid_width:
            self.vel = 0.0
            n_x = self.grid_width - self.width / 2.0

        self.center = (n_x, y)


class MyFruit():
    """
        Wrap the parameters and the dynamics related
        to the fruits.
    """

    def __init__(self, speed, size, grid_width, grid_height):
        """
        Set the parameters of the fruit (size, speed) and its initial position

        Arguments:
        ----------
        - `width`: Bar width
        - `height`: Bar height
        - `grid_width`: Grid width
        - `grid_height`: Grid height

        """

        self.speed = speed
        self.size = (size, size)

        self.grid_width = grid_width
        self.grid_height = grid_height

        # Force to get a self.center (valid)
        self.reset()

        """
        Defines ranges where the fruit can pop
        at the beginning of an episode and
        whenever it has been caught/miss by the bar
        """
        self.x_range = range(
            size * 2,
            self.grid_width - size * 2,
            size)
        self.y_range = range(
            size,
            int(self.grid_height / 2),
            size)

    def update(self, dt):
        """
        Updates the position of the fruit at a constant speed

        Arguments:
        ----------
        - `dt`: (single-step) integration constant

        """

        # Updates fruit position
        x, y = self.center
        n_y = y + self.speed * dt

        self.center = (x, n_y)

    def reset(self):
        """
            Resets to initial state, randomly somewhere in the grid,
            according to ranges defined by inner variables.
        """
        a, b = np.random.random((2,))
        x = np.floor((a * (self.grid_width - self.size[0])) + self.size[0] / 2.0)
        y = np.floor(b * ((self.grid_height - self.size[1]) / 2.0))

        self.center = (x, -1 * y)


class ContinuousCatcher():
    """
        Based on `Eder Santana's` game idea.
        (https://github.com/EderSantana)
    """

    @staticmethod
    def gamma():
        """
            Returns discount factor
        """
        return 0.95

    def __init__(self, width=640, height=640, init_lives=3, dt=30):
        """
        Wrapper for the full dynamics/parameters set of the game

        Arguments:
        ----------
        - `width`: Grid width
        - `height`: Grid height
        - `init_lives`: Number of allowed missed fruits before game over
        - `dt`: Frame per second (used as integration constant)

        """
        self.width = width
        self.height = height
        self.dt = dt
        self.fps = self.dt
        self.dx = 0.0
        self.init_lives = init_lives
        self.lives = init_lives

        # Parameters of the fruit
        self.fruit_size = percent_round_int(height, 0.06)
        self.fruit_fall_speed = 0.00095 * height

        # Parameters of the bar
        self.bar_speed = 0.021 * width
        self.bar_width = percent_round_int(width, 0.2)
        self.bar_height = percent_round_int(height, 0.04)

        # Reward function
        self.rtick = 1.0
        self.rpos = 2.0
        self.rneg = -2.0
        self.rloss = 0.0

        # Builds the bar with its parameters
        self.bar = MyBar(self.bar_width, self.bar_height,
                         self.width, self.height)
        self.bar_update = self.bar.update
        self.bar_reset = self.bar.reset

        # Builds the fruit with its parameters
        self.fruit = MyFruit(self.fruit_fall_speed, self.fruit_size,
                             self.width, self.height)
        self.fruit_update = self.fruit.update
        self.fruit_reset = self.fruit.reset

        self.nb_fruit_catch = 0

        self.history = list()
        self.total_reward = 0

    def reset(self):
        """
            Resets the game back to its initial state

            :return The observed state of the game
        """
        self.lives = self.init_lives
        self.fruit_reset()
        self.bar_reset()

        self.total_reward = 0

        self.history = list()
        self.nb_fruit_catch = 0
        return self.observe()

    def _collide_fruit(self):
        """
            Determines whether the bar hits the fruit

            :return True if the bar hits the fruit,
                    False otherwise
        """
        x1, y1 = self.bar.center
        x2, y2 = self.fruit.center
        w1, h1 = self.bar.size
        w2, h2 = self.fruit.size

        l1x, l1y = x1 - w1 / 2.0, y1 - h1 / 2.0
        l2x, l2y = x2 - w2 / 2.0, y2 - h2 / 2.0
        return (
                l1y < l2y + h2 and
                l2y < l1y + h1 and
                l1x < l2x + w2 and
                l2x < l1x + w1)

    def step(self, act):
        """
            Update the game with respect to its dynamics

            :param act array-like with only one dimension
                    act < 0 push left
                    act > 0 push right
                    act == 0 do nothing
        """
        done = False

        # Clip the absolute force to the maximum bar speed
        # Equivalent to : max(min(act[0], self.bar_speed),
        #                     -self.bar_speed)
        self.dx = np.clip(act[0], -self.bar_speed, self.bar_speed)

        # Grant reward related to tick and
        # whether fruit has been caught/missed
        reward = self.rtick
        if self.fruit.center[1] >= self.height:
            self.lives -= 1
            reward += self.rneg
            self.fruit_reset()

        if self._collide_fruit():
            self.nb_fruit_catch += 1
            self.fruit_reset()
            reward += self.rpos

        # Update bar and fruits
        self.bar_update(self.dx)
        self.fruit_update(self.fps)

        # Game over is reached when number of fruits have
        # trespassed a given thresold
        if self.lives == 0:
            reward += self.rloss
            done = True

        state = self.observe()

        self.history.append(np.hstack([state, reward]))
        self.total_reward += reward

        return state, reward, done

    def get_history(self):
        return np.array(self.history)

    def get_total_reward(self):
        return self.total_reward

    def observe(self):
        """
            Returns the current game state

            :return numpy array of the form:
                        [bar_center_x, bar_velocity, fruit_center_x, fruit_center_y]
        """
        return np.asarray([self.bar.center[0], self.bar.vel,
                           self.fruit.center[0], self.fruit.center[1]])
