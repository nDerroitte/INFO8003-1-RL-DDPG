import numpy as np
import math
from keras.layers import Dense, Input, Concatenate
from keras.models import  Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, BATCH_SIZE, TAU, GAMMA):
        """
        Parameters:
        -----------
        sess : A tensorflow session
        state_dim : int
            Number of dimension of the state-space
        action_dim : int
            Number of dimension of the action-space
        BATCH_SIZE: int
            Size of the batcj
        TAU : double
            Learning rate of the target network
        GAMMA: double
            Learning rate of the AdamOptimizer
        """
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.state_dim = state_dim
        self.action_dim = action_dim

        K.set_session(sess)

        self.model, self.action, self.state = self.__create_model()
        self.target_model, _, _ = self.__create_model()
        self.action_grads = tf.gradients(self.model.output, self.action)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def gradients(self, states, actions):
        """
        Returns:
        --------
        The gradients of critic network
        """
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        """
        Simple call of the predict function of the keras model for the target
        network
        Parameters:
        -----------
        states : []
            List of the 4 elements composing the states
        """
        weights_model = self.model.get_weights()
        weights_target = self.target_model.get_weights()
        for i in range(len(weights_model)):
            weights_target[i] = self.TAU * weights_model[i] + (
                                       (1 - self.TAU) * (weights_target[i]))
        self.target_model.set_weights(weights_target)

    def __create_model(self):
        """
        Create the actor model

        Returns:
        ---------
        keras Model : the actor network
        Action Input
        State Input
        """
        S = Input(shape=[self.state_dim])
        A = Input(shape=[self.action_dim],name='action2')

        w1 = Dense(512, activation='relu')(S)
        a1 = Dense(512, activation='linear')(A)
        h1 = Dense(256, activation='linear')(w1)

        h2 =  Concatenate(axis=-1)([h1, a1])

        h3 = Dense(128, activation='relu')(h2)
        V = Dense(self.action_dim,activation='relu')(h3)

        model = Model(inputs=[S,A],outputs=V)
        adam = Adam(lr=self.GAMMA)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S
