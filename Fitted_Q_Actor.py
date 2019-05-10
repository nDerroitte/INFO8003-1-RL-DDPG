from keras.layers import Dense, Dropout, Input, Lambda
from keras.models import Model, Sequential
from keras.layers.merge import Concatenate, Multiply
from keras.optimizers import Adam

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor


import numpy as np
import random

class Fitted_Q_Actor():
    def __init__(self, agent, epsilon, model_name, gamma, N, epsilon_decay=None, load_model=False,
                 learning_rate=0.001):

        self.agent = agent

        self.N = N

        self.learning_rate = learning_rate

        self.epsilon = epsilon

        self.epsilon_decay = epsilon_decay

        self.gamma = gamma

        self.model_name = model_name

        self.actor_model = self.get_q_model(model_name)

        self.already_fitted = False

        self.history = list()

        if load_model is True:
            self.load_models()

    def get_q_model(self, model_name):
        if model_name == "Linear Regression":
            return SGDRegressor()

        elif model_name == "Extremely Randomized Trees":
            return ExtraTreesRegressor(n_estimators=100, bootstrap=True, warm_start=True, max_features=None)

        elif model_name == "Neural Networks":
            activation_function = "relu"

            state_input = Input(shape=(self.agent.get_state_space_dim(),))
            state_h1 = Dense(24, activation=activation_function)(state_input)
            state_h2 = Dense(48)(state_h1)

            action_input = Input(shape=(self.agent.get_action_space_dim(),))
            action_h1 = Dense(48)(action_input)

            merged = Concatenate(axis=-1)([state_h2, action_h1])
            merged_h1 = Dense(24, activation=activation_function)(merged)

            # Get a Value between -1 and 1
            restricted_output = Dense(self.agent.get_action_space_dim(), activation='tanh')(merged_h1)

            # Get an output between -FORCE_MAX and FORCE_MAX
            output = Lambda(lambda x: x * self.agent.get_max_abs_output())(restricted_output)

            model = Model(inputs=[state_input, action_input], outputs=output)

            adam = Adam(self.learning_rate)
            model.compile(loss="mse", optimizer=adam)

            return model

        else:
            raise Exception(("Fitted Q-Learning not implemented with " + model_name + " algorithm."))

    def train(self, agent_history):
        self.history.append(agent_history)

        self.compute_online_fitted_q()

    def compute_online_fitted_q(self):
        states, actions, next_states, rewards = self.pretreat_history([self.history[-1]])

        for i in range(1, self.N + 1):

            print("Computing Q" + str(i), "with Fitted Q-Iteration and a", self.model_name, "model ...", end='\r')

            training_sample, labels = self.get_fitted_q_training_sample(states, actions, next_states, rewards)

            if len(training_sample) == 0:
                print("")
                print("Error: The history must be non empty for the fitted-q iteration algorithm to work.",
                      file=sys.stderr)
                exit(-1)

            self.train_q_model(training_sample, labels)

            self.already_fitted = True

    def compute_fitted_q(self):
        self.actor_model = self.get_q_model(self.model_name)

        self.already_fitted = False

        states, actions, next_states, rewards = self.pretreat_history(self.history)

        for i in range(1, self.N + 1):
            print("Computing Q" + str(i), "with Fitted Q-Iteration and a", model_name, "model ...", end='\r')

            training_sample, labels = self.get_fitted_q_training_sample(states, actions, next_states, rewards)

            if len(training_sample) == 0:
                print("")
                print("Error: The history must be non empty for the fitted-q iteration algorithm to work.",
                      file=sys.stderr)
                exit(-1)

            self.train_q_model(training_sample, labels)

            self.already_fitted = True

    def pretreat_history(self, h):
        states = list()
        actions = list()
        rewards = list()
        next_states = list()

        for trajectory in h:
            for bar_center, bar_speed, fruit_center_x, fruit_center_y, action, reward in trajectory[0:-1]:
                states.append([bar_center, bar_speed, fruit_center_x, fruit_center_y])
                actions.append(action)

                next_state = self.agent.observe_next_state \
                    ([bar_center, bar_speed, fruit_center_x, fruit_center_y, action], [action])

                next_states.append([next_state[0], next_state[1], next_state[2], next_state[3]])

                rewards.append(reward)

        if len(states) == 0:
            bar_center, bar_speed, fruit_center_x, fruit_center_y = self.agent.get_init_state()

            action = 0

            states.append([bar_center, bar_speed, fruit_center_x, fruit_center_y])
            actions.append(action)

            next_state = self.agent.observe_next_state \
                ([bar_center, bar_speed, fruit_center_x, fruit_center_y, action], [action])
            next_states.append(next_state)

            rewards.append(1)

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1)

    def get_fitted_q_training_sample(self, states, actions, next_states, rewards):
        if self.already_fitted is True:
            next_state_best_actions, next_state_max_q = self.compute_max_q(next_states, self.actor_model)

            Y = np.array(rewards).reshape(-1, 1) + self.gamma * np.array(next_state_max_q).reshape(-1, 1)

        else:
            Y = np.array(rewards)

        if self.model_name == "Linear Regression":
            X = np.hstack([states, actions])

        elif self.model_name == "Extremely Randomized Trees":
            X = np.hstack([states, actions])

        elif self.model_name == "Neural Networks":
            X = [states, actions]

        else:
            raise Exception("Error: Unimplemented training samples for " + str(self.model_name) + " model.")

        return X, np.array(Y).ravel()

    def train_q_model(self, training_sample, labels):
        if self.model_name == "Linear Regression":
            self.actor_model.partial_fit(training_sample, labels)

        elif self.model_name == "Extremely Randomized Trees":
            model.n_estimators = model.n_estimators + 10
            self.actor_model.fit(training_sample, labels)

        elif self.model_name == "Neural Networks":
            self.actor_model.fit(training_sample, labels, verbose=0)

        else:
            raise Exception(("Fitted Q-Learning not implemented with " + model_name + " algorithm."))

    def compute_max_q(self, states, q):
        best_actions = list()

        for state in states:
            # Initialisation des paramètres
            mu, sigma = self.cross_entropy_max_q(state, q)

            best_actions.append(mu)

        best_q = q.predict([np.array(states), np.array(best_actions).reshape(-1, 1)])

        return best_actions, best_q

    def cross_entropy_max_q(self, state, q, N=100, Ne=70, epsilon=1e-3):
        """
        :param state:
        :param q:
        :param N: Number of actions that will be considered at each step as an approximator of the distribution of Q
                    for a given particular state
        :param Ne:
        :param epsilon:
        :return:
        """
        # Initialisation des paramètres
        mu = 0
        sigma = 1

        while sigma > epsilon:  # Tant que l'on n'a pas convergé

            actions = np.array([random.gauss(mu, sigma) for i in range(0, N)]).reshape(-1, 1)

            states = np.array([state for i in range(0, N)])

            S = q.predict([states, actions])

            X = np.array(sorted(zip(S, actions)))[:, 1]  # Classe X selon le score(de façon descendante)

            mu = np.mean(X[0:Ne])

            sigma = np.var(X[0:Ne])  # Mise à jour des paramètres de la distribution

        return mu, sigma

    def act(self, state):
        """
            Exploration - Exploitation Choice
        """
        e = random.random()

        # Exploration
        if e < self.epsilon:
            move = self.agent.random_move()
        # Exploitation
        else:
            move, best_reward = self.compute_max_q([state], self.actor_model)

        return move

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def apply_experience_replay(self, history):
        nb_replay = random.randint(0, len(history) / 2)

        replay_list = list()

        for i in range(0, nb_replay):
            replay_list.append(random.choice(history[-1]))

        return list(history) + replay_list

    def store_models(self):
        if self.model_name == "Neural Networks":
            # serialize model to JSON
            actor_model_json = self.actor_model.to_json()

            with open("fitted_q_actor_model.json", "w") as json_file:
                json_file.write(actor_model_json)

            # serialize weights to HDF5
            self.actor_model.save_weights("fitted_q_actor_model.h5")

            print("Saved Actor-Critic model to disk")

    def load_models(self):

        if self.model_name == "Neural Networks":
            # load json and create model
            actor_model_json_file = open('fitted_q_actor_model.json', 'r')

            actor_model_json = actor_model_json_file.read()

            actor_model_json_file.close()

            self.actor_model = model_from_json(actor_model_json)

            # load weights into new model
            self.actor_model.load_weights("fitted_q_actor_model.h5")

            adam = Adam(lr=self.learning_rate)
            self.actor_model.compile(loss="mse", optimizer=adam)

            print("Loaded Actor-Critic model from disk")