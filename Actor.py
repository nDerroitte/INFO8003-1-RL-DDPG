import numpy as np
from sklearn.tree import ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import random
<<<<<<< HEAD
from keras.layers import Dense, Activation, Input, Concatenate
from keras.models import Sequential, Model
from keras import optimizers
import keras

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, GAMMA, TAU,states, actions,rewards):
        self.__action_dim = action_dim
        self.LEARNING_RATE = 0.001
        self.gamma = GAMMA
        self.epsilon = 1
        self.counter = 0
        self.model = self.__create_model()
        print(actions)
        self.model.train_on_batch([states, actions], rewards)
        


    # def train(self, states, rewards, next_states):
    #     # self.model.fit(states, Q_values, epochs=150, batch_size=10)
    #     # Q_model = KNeighborsRegressor(n_neighbors=1)
    #     # Q_model.fit(states,rewards)
    #     #print(states)
    #     next_Q_values = self.model.predict(next_states)
    #     # next_Q_values = self.model.predict(next_states)
    #     for i in range(len(rewards)):
    #         rewards[i] = rewards[i] + self.gamma*next_Q_values[i]
    #     # for i in range(len(rewards)):
    #     #     rewards[i] = rewards[i]+self.gamma*self.model.predict(states[i,:].reshape(1,-1))
    #     print(len(states))
    #     self.model.fit(states,rewards.reshape(-1,1),epochs=1, batch_size=len(states))
    #     # self.model.fit(states,rewards)

    def train_batch(self, states, actions, rewards, next_states,action_list):
        Q_values_next_step = np.zeros((len(next_states),len(action_list)))
        for i in range(len(next_states)):
            for j in range(len(action_list)):
                Q_values_next_step[i,j] = self.model.predict([[next_states[i]],np.array([action_list[j]])])
                # Q_values_next_step[i,j] = self.model.predict(np.array([[next_states[i][0],next_states[i][1],next_states[i][2],next_states[i][3],action_list[j]]]))
        # for i in range(len(next_states)):
        #     act = np.full((len(next_states),),action_list[i])
        #     print(act)
        #     Q_values_next_step[:,i] = self.model.predict([next_states,act])
        Q_values = rewards + self.gamma * np.max(Q_values_next_step, axis=1)
        
        if self.counter%5000 == 0:
            print("next : ",Q_values_next_step)
            print("Q values : ",Q_values)
        
        self.counter += 1
        if self.counter%5000 == 0:
            print(self.counter)
        self.model.train_on_batch([states,actions], Q_values)
        self.model.save_weights('DQN.h5')


    def predict(self, state, action_list):
        if self.counter < 13500:
            self.epsilon = 1-self.counter/15000
        else:
            self.epsilon = 0.05
        
        if random.random() < self.epsilon:
            action = action_list[random.randint(0,len(action_list)-1)]
        else:
            action = action_list[0]
            score = 0
            state.reshape(1,-1)
            for el in action_list:
                input_data = np.concatenate((state,[el])).reshape(1,-1)
                #print(input_data)
                # predicted_score = self.model.predict(input_data)
                predicted_score = self.model.predict([[state],np.array([el])])
                #print(el,"prediction :",predicted_score,self.epsilon)
                if predicted_score > score:
                    action = el
                    score = predicted_score
        
=======
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
>>>>>>> 103ae2d33a992523b0816c93d1ae5c54f79bac47
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
<<<<<<< HEAD
            # model.add(Dense(128, input_shape=(5,), activation='relu'))
            # model.add(Dense(128))
            # model.add(Dense(1, activation='relu'))

            # # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            # # model.compile(loss='mean_squared_error', optimizer=sgd)

            # optimizer = optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
            # # optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            # model.compile(optimizer, loss='mse')
            S = Input(shape=(4,))
            A = Input(shape=(1,),name='action2')

            w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
            a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
            h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)

            h2 =  Concatenate(axis=-1)([h1, a1])

            h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
            V = Dense(1,activation='relu')(h3)

            model = Model(inputs=[S,A],outputs=V)
            adam = optimizers.Adam(lr=self.LEARNING_RATE)
            model.compile(loss='mse', optimizer=adam)
            return model
            # return ExtraTreeRegressor()
=======
            # model.add(Dense(16, input_shape=(4,), activation='relu'))
            # model.add(Dense(32))
            # model.add(Activation('relu'))

            # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            # model.compile(loss='mean_squared_error', optimizer=sgd)

            # return model
            return LinearRegression()
>>>>>>> 103ae2d33a992523b0816c93d1ae5c54f79bac47
        except ValueError as e:
            print("""Error: couldn't create the
                     actor network :{}.""".format(e))
