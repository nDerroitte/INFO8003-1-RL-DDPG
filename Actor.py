import numpy as np
from sklearn.tree import ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import random
#from keras.models import Sequential, Dense, Activation, optimizers

class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, GAMMA, TAU,states,rewards):
        self.__action_dim = action_dim
        self.model = self.__create_model()
        self.model.fit(states,rewards)
        self.gamma = GAMMA


    def train(self, states, rewards, next_states):
        # self.model.fit(states, Q_values, epochs=150, batch_size=10)
        # Q_model = KNeighborsRegressor(n_neighbors=1)
        # Q_model.fit(states,rewards)
        #print(states)
        next_Q_values = self.model.predict(states)
        rewards = rewards + self.gamma*next_Q_values
        # for i in range(len(rewards)):
        #     rewards[i] = rewards[i]+self.gamma*self.model.predict(states[i,:].reshape(1,-1))
        self.model.fit(states,rewards)


    def predict(self, state, action_list):
        action = action_list[2]
        score = 0
        state.reshape(1,-1)
        for el in action_list:
            input_data = np.concatenate((state,[el])).reshape(1,-1)
            #print(input_data)
            predicted_score = self.model.predict(input_data)*random.uniform(1,1.03)
            if predicted_score > score:
                action = el
                score = predicted_score
        if random.random() > 0.95:
            tmp_action_list = list(action_list)
            tmp_action_list.remove(action)
            action = tmp_action_list[random.randint(0,3)]
        return action

    def target_predict(self, states):
        return 0

    def target_train(self):
        # weights_model = self.model.get_weights()
        # weights_target = self.target_model.get_weights()

        # for i in range(len(weights_model)):
        #     weights_target[i] = (self.__TAU * weights_model[i]) + (
        #                         (1 - self.__TAU) * weights_target[i])
        # self.target_model.set_weights(weights_target)
        return 0

    def __create_model(self):
        try:
            # model = Sequential()
            # model.add(Dense(16, input_shape=(4,), activation='relu'))
            # model.add(Dense(32))
            # model.add(Activation('relu'))

            # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            # model.compile(loss='mean_squared_error', optimizer=sgd)

            # return model
            return LinearRegression()
        except ValueError as e:
            print("""Error: couldn't create the
                     actor network :{}.""".format(e.strerror))
