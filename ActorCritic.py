import numpy as np
import tensorflow as tf
import random
from collections import deque

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers.merge import Concatenate, Multiply
from keras.optimizers import Adam
from keras.models import model_from_json

ACTOR_MODEL = ["Neural Network 1"]

CRITIC_MODEL = ["Neural Network 1"]


class ActorCritic:

    def __init__(self, agent, epsilon, actor_model_name, critic_model_name, gamma, epsilon_decay=None,
                 load_model=False, learning_rate=0.001):

        # Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        self.agent = agent

        self.learning_rate = learning_rate

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.gamma = gamma

        self.memory = deque(maxlen=2000)

        self.actor_model_name = actor_model_name
        self.critic_model_name = critic_model_name

        self.actor_state_input = None
        self.actor_model = None
        self.target_actor_model = None
        self.critic_state_input = None
        self.critic_action_input = None
        self.critic_model = None
        self.target_critic_model = None
        self.actor_critic_grad = None


        if load_model is True:
            self.load_models()

            self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.agent.get_action_space_dim()])

            actor_model_weights = self.actor_model.trainable_weights

            self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, - self.actor_critic_grad)

            grads = zip(self.actor_grads, actor_model_weights)
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

            self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        else:
            """
                    ACTOR MODEL - This model will decide from a state which action to take
            """

            self.actor_state_input, self.actor_model = self.create_actor_model(actor_model_name)
            _, self.target_actor_model = self.create_actor_model(actor_model_name)

            self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.agent.get_action_space_dim()])

            actor_model_weights = self.actor_model.trainable_weights

            self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights,
                                            -self.actor_critic_grad)  # dC/dA (from actor)

            grads = zip(self.actor_grads, actor_model_weights)
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

            """
                    CRITIC MODEL - This model will evaluate the decision of the actor model
            """

            self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model(
                critic_model_name)
            _, _, self.target_critic_model = self.create_critic_model(critic_model_name)

            self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        self.sess.run(tf.global_variables_initializer())

    def create_actor_model(self, model_name):
        global ACTOR_MODEL

        if model_name == ACTOR_MODEL[0]:
            activation_function = "relu"

            state_input = Input(shape=(self.agent.get_state_space_dim(),))

            h1 = Dense(24, activation=activation_function)(state_input)
            h2 = Dense(48, activation=activation_function)(h1)
            h3 = Dense(24, activation=activation_function)(h2)

            # Get a Value between -1 and 1
            restricted_output = Dense(self.agent.get_action_space_dim(), activation='tanh')(h3)

            # Get an output between -FORCE_MAX and FORCE_MAX
            output = Lambda(lambda x: x * self.agent.get_max_abs_output())(restricted_output)

            model = Model(inputs=state_input, outputs=output)
            adam = Adam(self.learning_rate)
            model.compile(loss="mse", optimizer=adam)

            return state_input, model

        else:
            raise Exception("Error: Unimplemented actor model " + str(model_name) + ".")

    def create_critic_model(self, model_name):
        global CRITIC_MODEL

        if model_name == CRITIC_MODEL[0]:
            activation_function = "relu"

            state_input = Input(shape=(self.agent.get_state_space_dim(),))
            state_h1 = Dense(24, activation=activation_function)(state_input)
            state_h2 = Dense(48)(state_h1)

            action_input = Input(shape=(self.agent.get_action_space_dim(),))
            action_h1 = Dense(48)(action_input)

            merged =  Concatenate(axis=-1)([state_h2, action_h1])
            merged_h1 = Dense(24, activation=activation_function)(merged)

            output = Dense(1, activation="relu")(merged_h1)
            model = Model(inputs=[state_input, action_input], outputs=output)

            adam = Adam(self.learning_rate)
            model.compile(loss="mse", optimizer=adam)

            return state_input, action_input, model

        else:
            raise Exception("Error: Unimplemented critic model " + str(model_name) + ".")

    def remember(self, curr_state, action, reward, new_state, done):
        self.memory.append([curr_state, action, reward, new_state, done])

    def train_actor(self, samples):
        for curr_state, move, reward, next_state, done in samples:
            predicted_action = self.actor_model.predict(curr_state)

            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: curr_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: curr_state,
                self.actor_critic_grad: grads
            })

    def train_critic(self, samples):
        for curr_state, move, reward, next_state, done in samples:
            if not done:
                target_action = self.target_actor_model.predict(next_state)

                future_reward = self.target_critic_model.predict(
                    [next_state, target_action])[0][0]

                reward += self.gamma * future_reward

            self.critic_model.fit([np.array(curr_state), np.array([move])],
                                  np.array([reward]), verbose=0)

    def train(self, curr_state, move, reward, next_state, done, batch_size=32):
        self.remember(curr_state, move, reward, next_state, done)

        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        self.train_critic(samples)
        self.train_actor(samples)

    def update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]

        self.target_critic_model.set_weights(actor_target_weights)

    def update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]

        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self.update_actor_target()
        self.update_critic_target()

    def act(self, curr_state):
        if np.random.random() < self.epsilon:
            return self.agent.random_move()

        else:
            return self.actor_model.predict(curr_state).squeeze()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def store_models(self):
        # serialize weights to HDF5
        self.actor_model.save_weights("actor_model.h5")
        self.critic_model.save_weights("critic_model.h5")

        print("Saved Actor-Critic model to disk")

    def load_models(self):
        self.actor_state_input, self.actor_model = self.create_actor_model(self.actor_model_name)
        _, self.target_actor_model = self.create_actor_model(self.actor_model_name)


        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model(
            self.critic_model_name)
        _, _, self.target_critic_model = self.create_critic_model(self.critic_model_name)

        # load weights into new model

        self.actor_model.load_weights("actor_model.h5")
        self.target_actor_model.load_weights("actor_model.h5")

        self.critic_model.load_weights("critic_model.h5")
        self.target_critic_model.load_weights("critic_model.h5")

        adam = Adam(lr=self.learning_rate)

        self.actor_model.compile(loss="mse", optimizer=adam)
        self.target_actor_model.compile(loss="mse", optimizer=adam)

        self.critic_model.compile(loss="mse", optimizer=adam)
        self.target_critic_model.compile(loss="mse", optimizer=adam)

        print("Loaded Actor-Critic model from disk")
