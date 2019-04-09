import random
import sys

from keras.layers import Dense
from keras.models import Sequential
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor

from RBF_Layer import *
from Util import *
from catcher import ContinuousCatcher
from gui import *


class Simulation:
    def __init__(self, gamma):
        self.agent = ContinuousCatcher()

        self.gamma = gamma

        self.already_fitted = False

    def run_greedy(self, nb_episode, episode_size, N, epsilon, model_name, exploration_decay=None,
                   experience_replay=False, alpha=None, display=False):
        h = list()
        final_rewards = list()

        if alpha is None:
            q = self.get_q_model(model_name)
        else:
            q = self.get_q_learning_approximator_model(model_name)

        self.already_fitted = False

        e = epsilon

        self.agent.lives = episode_size

        gui = GUI()

        print("Running Simulation with an e-greedy policy and a " + model_name + " model :")
        for i in range(1, nb_episode + 1):

            agent_history = self.agent.get_history()
            self.agent.reset()
            gui.updateGUI(self.agent, i)

            if experience_replay is False:
                h.append(agent_history)
            else:
                replayed = self.apply_experience_replay(agent_history, nb_replay=10)
                h.append(replayed)

            if alpha is None:
                print("Episode ", i, " : Fitted-Q Learning computation...")
                self.compute_online_fitted_q(h, N, q, model_name)
            else:
                print("Episode ", i, " : Parametric Q Learning computation...")
                self.compute_online_q_learning_approximation(h, N, q, alpha, model_name)

            done = False
            while done is False:
                state, reward, done = self.update_greedy(q, e)

                gui.updateGUI(self.agent, i)

            total_reward = self.agent.get_total_reward()
            final_rewards.append(total_reward)

            if display is True:
                print("Episode ", str(i), " : Total Reward = ", str(self.agent.get_total_reward()))
                print("Episode ", str(i), " : Last State = ", str(self.agent.observe()))

            if exploration_decay is not None:
                e = e - exploration_decay * i

        gui.closeGUI()

        return final_rewards, q

    def run_softmax(self, nb_episode, episode_size, N, temperature, model_name,
                    experience_replay=False, alpha=None, display=False):
        raise Exception("Error : Not Implemented Policy")

        h = list()
        final_rewards = list()

        if alpha is None:
            q = self.get_q_model(model_name)
        else:
            q = self.get_q_learning_approximator_model(model_name)

        self.already_fitted = False

        self.agent.lives = episode_size

        print("Running Simulation with a Softmax policy  and a " + model_name + " model :")

        for i in range(1, nb_episode + 1):

            agent_history = self.agent.get_history()
            self.agent.reset()

            if experience_replay is False:
                h.append(agent_history)
            else:
                replayed = self.apply_experience_replay(agent_history, nb_replay=10)
                h.append(replayed)

            if alpha is None:
                print("Episode ", i, " : Fitted-Q Learning computation...")
                self.compute_online_fitted_q(h, N, q, model_name)
            else:
                print("Episode ", i, " : Parametric Q Learning computation...")
                self.compute_online_q_learning_approximation(h, N, q, alpha, model_name)

            move_proba = self.compute_softmax_move_proba(q, temperature)

            while True:
                if self.update_softmax(move_proba) is True:
                    break

            total_reward = self.agent.get_total_reward()
            final_rewards.append(total_reward)

            if display is True:
                print("Episode ", str(i), " : Total Reward = ", str(self.agent.get_total_reward()))
                print("Episode ", str(i), " : Last State = ", str(self.agent.observe()))

        return final_rewards, q

    def compute_online_fitted_q(self, h, n, q, model_name):
        X, next_states, rewards = self.pretreat_history([h[-1]])
        for i in range(1, n + 1):
            training_sample, labels = self.get_fitted_q_training_sample(q, X, next_states, rewards)

            if len(training_sample) == 0:
                print("Error: The history must be non empty for the fitted-q iteration algorithm to work.",
                      file=sys.stderr)
                exit(-1)

            self.train_q_model(training_sample, labels, q, model_name)

            self.already_fitted = True

    def compute_online_q_learning_approximation(self, h, n, q, alpha, model_name):
        X, next_states, rewards = self.pretreat_history([h[-1]])
        for i in range(1, n + 1):
            training_sample, labels = self.get_q_learning_training_sample(q, X, next_states, rewards, alpha)

            if len(training_sample) == 0:
                print("Error: The history must be non empty for the fitted-q iteration algorithm to work.",
                      file=sys.stderr)
                exit(-1)

            self.train_q_learning_approximator(training_sample, labels, q, model_name)

            self.already_fitted = True

    def compute_fitted_q(self, h, n, model_name, display=False):
        q = self.get_q_model(model_name)

        self.already_fitted = False

        X, next_states, rewards = self.pretreat_history(h)

        for i in range(1, n + 1):
            if display is True:
                print("Computing Q" + str(i), "with Fitted Q-Iteration and a", model_name, "model ...", end='\r')

            training_sample, labels = self.get_fitted_q_training_sample(q, X, next_states, rewards)

            if len(training_sample) == 0:
                print("Error: The history must be non empty for the fitted-q iteration algorithm to work.",
                      file=sys.stderr)
                exit(-1)

            self.train_q_model(training_sample, labels, q, model_name)

            self.already_fitted = True

        print("")
        return q

    def compute_q_learning_approximation(self, h, n, alpha, model_name, display=False):
        q = self.get_q_learning_approximator_model(model_name)

        self.already_fitted = False

        X, next_states, rewards = self.pretreat_history([h])

        for i in range(1, n + 1):
            if display is True:
                print("Computing Q" + str(i), "with Parametric Q-Learning and a", model_name, "model ...", end='\r')

            training_sample, labels = self.get_q_learning_training_sample(q, X, next_states, rewards, alpha)

            if len(training_sample) == 0:
                print("Error: The history must be non empty for the fitted-q iteration algorithm to work.",
                      file=sys.stderr)
                exit(-1)

            self.train_q_learning_approximator(training_sample, labels, q, model_name)

            self.already_fitted = True

        print("")

        return q

    def pretreat_history(self, h):
        X = list()
        rewards = list()
        next_states = list()

        for trajectory in h:
            for bar_center, bar_speed, fruit_center_x, fruit_center_y, action, reward in trajectory[0:-1]:
                X.append([bar_center, bar_speed, fruit_center_x, fruit_center_y, action])

                next_state = self.agent.observe_next_state \
                    ([bar_center, bar_speed, fruit_center_x, fruit_center_y, action], [action])

                next_states.append([next_state[0], next_state[1], next_state[2], next_state[3]])

                rewards.append(reward)

        if len(X) == 0:
            bar_center, bar_speed, fruit_center_x, fruit_center_y = self.agent.get_init_state()

            action = 0

            X.append([bar_center, bar_speed, fruit_center_x, fruit_center_y, action])

            next_state = self.agent.observe_next_state \
                ([bar_center, bar_speed, fruit_center_x, fruit_center_y, action], [action])
            next_states.append(next_state)

            rewards.append(1)

        return np.array(X), np.array(next_states), np.array(rewards).reshape(-1, 1)

    def get_fitted_q_training_sample(self, q, X, next_states, rewards):
        Y = list()
        if self.already_fitted is True:
            next_state_best_actions, next_state_max_q = self.cross_entropy_max_q(next_states, q)

            k = 0
            while k < len(next_state_max_q):
                Y.append(rewards[k] + self.gamma * next_state_max_q[k])

                k = k + 1
        else:
            for reward in rewards:
                Y.append(reward)

        return X, np.array(Y).ravel()

    def get_q_learning_training_sample(self, q, X, next_states, rewards, alpha):
        Y = list()
        if self.already_fitted is True:
            next_state_best_actions, next_state_max_q = self.cross_entropy_max_q(next_states, q)

            curr_q = q.predict(X)

            Y = list()

            k = 0
            while k < len(next_state_max_q):
                temp_diff = rewards[k] + self.gamma * next_state_max_q[k] - curr_q[k]

                Y.append(curr_q[k] + alpha * temp_diff)
                k = k + 1

        else:
            for reward in rewards:
                Y.append(alpha * reward)

        return X, np.array(Y).ravel()

    def cross_entropy_max_q(self, states, q, N=1000, Ne=100, epsilon=1e-3):
        best_actions = list()

        for state in states:
            # Initialisation des paramètres
            mu = 0
            sigma = 1

            while sigma > epsilon:  # Tant que l'on n'a pas convergé

                actions = np.array([random.gauss(mu, sigma) for i in range(0, N)]).reshape(-1, 1)

                tmp = np.hstack([np.array([state for i in range(0, N)]), actions])

                S = q.predict(tmp)

                X = np.array(sorted(zip(S, actions)))[:, 1]  # Classe X selon le score(de façon descendante)

                mu = np.mean(X[1:Ne])

                sigma = np.var(X[1:Ne])  # Mise à jour des paramètres de la distribution

            best_actions.append(mu)

        tmp = np.hstack([np.array(states), np.array(best_actions).reshape(-1, 1)])

        best_q = q.predict(tmp)

        return best_actions, best_q

    def train_q_model(self, training_sample, labels, model, model_name):
        if model_name == "Linear Regression":
            model.partial_fit(training_sample, labels)

        elif model_name == "Extremely Randomized Trees":
            model.n_estimators = model.n_estimators + 10
            model.fit(training_sample, labels)

        elif model_name == "Neural Networks":
            model.fit(training_sample, labels, verbose=0)

        else:
            raise Exception(("Fitted Q-Learning not implemented with " + model_name + " algorithm."))

    def train_q_learning_approximator(self, training_sample, labels, model, model_name):
        if model_name == "Neural Networks":
            model.fit(training_sample, labels, verbose=0)

        elif model_name == "Radial Basis":
            model.fit(training_sample, labels, verbose=0)

        else:
            raise Exception(("Fitted Q-Learning not implemented with " + model_name + " algorithm."))

    def get_q_model(self, model_name):
        if model_name == "Linear Regression":
            return SGDRegressor()

        elif model_name == "Extremely Randomized Trees":
            return ExtraTreesRegressor(n_estimators=100, bootstrap=True, warm_start=True, max_features=None)

        elif model_name == "Neural Networks":

            model = Sequential()

            model.add(Dense(16, input_dim=5, activation='relu'))

            model.add(Dense(1, activation='relu'))

            model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])

            return model

        else:
            raise Exception(("Fitted Q-Learning not implemented with " + model_name + " algorithm."))

    def get_q_learning_approximator_model(self, model_name):
        if model_name == "Neural Networks":

            model = Sequential()

            model.add(Dense(16, input_dim=5, activation='relu'))

            model.add(Dense(1, activation='relu'))

            model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])

            return model

        elif model_name == "Radial Basis":
            model = Sequential()

            model.add(Dense(16, input_dim=5, activation='relu'))

            model.add(RBFLayer(1, 0.5))

            model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])

            return model

        else:
            raise Exception(("Q-Learning Approximator not implemented with " + model_name + " algorithm."))

    def get_next_state(self, q, state=None):
        """
        This function will return the state that would be chosen by the
        agent in its current state or in the state mentioned
        (if one is mentioned) according to the chosen policy.

        """

        if state is None:
            state = self.agent.observe()
        else:
            state = state

        move, best_q = self.cross_entropy_max_q([state], q)

        next_state = self.agent.observe_next_state(state, move)

        return move, next_state

    def update_greedy(self, q, epsilon):
        state = self.agent.observe()

        """
            Exploration - Exploitation Choice
        """
        e = np.random.rand()

        # Exploration
        if e < epsilon:
            move = random.random()
        # Exploitation
        else:
            best_move, best_reward = self.cross_entropy_max_q([state], q)

            move = best_move

        return self.agent.step(move)

    def get_greedy_update(self, state, q, epsilon):
        """
            Exploration - Exploitation Choice
        """

        """
            Exploration - Exploitation Choice
        """
        e = np.random.rand()

        # Exploration
        if e < epsilon:
            move = random.random()
        # Exploitation
        else:
            best_move, best_reward = self.cross_entropy_max_q([state], q)

            move = best_move

        next_state = self.agent.observe_next_state(state, move)

        return next_state, move

    def update_softmax(self, move_proba):
        """
            Exploration - Exploitation Choice
        """
        j = 0
        a = dict()
        for i in move_proba.keys():
            a[j] = i
            j = j + 1

        choice = np.random.choice(np.array(list(a.keys())), p=list(move_proba.values()))
        move = a[choice]

        state = self.agent.observe()
        dynamic = self.agent.get_dynamic(state)

        next_pos = dynamic[move]

        self.agent.move(next_pos, move)

    def compute_softmax_move_proba(self, q, temperature):
        move_proba = None
        if q is not None:
            state = self.agent.observe()

            # Compute Probabilities
            move_proba = dict()
            denom = 0
            for move in self.action_space:
                tmp = np.array([state[0], state[1], move]).reshape(1, -1)
                q_t = q.predict(tmp)[0]
                tmp = q_t / temperature
                move_proba[move] = math.exp(tmp)

                denom = denom + move_proba[move]

            for i in move_proba:
                move_proba[i] = move_proba[i] / denom

        return move_proba

    def apply_experience_replay(self, history, nb_replay):
        already_replayed = 0

        while already_replayed < nb_replay:
            last_state = history[-1][0]
            replay = False
            for state, action, reward in history:
                if state == last_state:
                    history[-1][1] = action
                    history[-1][2] = reward
                    replay = True
            if replay is not True:
                break

            already_replayed = already_replayed + 1

        return history
