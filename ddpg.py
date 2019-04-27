from ReplayBuffer import *
import numpy as np
from OU import *

BUFFER_SIZE = 100000
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.001     #Target Network HyperParameters
LRA = 0.0001    #Learning rate for Actor
LRC = 0.001     #Lerning rate for Critic
epsilon = 1
action_dim = 1  # Move
state_dim = 4 # Bar : x,vx Fruit : x,y
OU = OU()       #Ornstein-Uhlenbeck Process


def playGame(sess, states, actor, train_indicator):    #1 means Train, 0 means simply Run
    a_t = np.zeros([1,action_dim])
    noise_t = np.zeros([1,action_dim])

    a_t_original = actor.model.predict(states.reshape(1, states.shape[0]))
    noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
    a_t[0][0] = a_t_original[0][0] + noise_t[0][0]

    return a_t[0]

def trainModels(buff, sess, st, r, st1, a, loss, actor, critic, train_indicator):
    buff.add(st, a, r, st1)

    batch = buff.getBatch(BATCH_SIZE)
    states = np.asarray([e[0] for e in batch])
    actions = np.asarray([e[1] for e in batch])
    rewards = np.asarray([e[2] for e in batch])
    new_states = np.asarray([e[3] for e in batch])

    y_t = np.asarray([e[1] for e in batch])

    target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

    for k in range(len(batch)):
        y_t[k] = rewards[k] + GAMMA*target_q_values[k]

    if train_indicator:
        loss += critic.model.train_on_batch([states,actions], y_t)
        a_for_grad = actor.model.predict(states)
        grads = critic.gradients(states, a_for_grad)
        actor.train(states, grads)
        actor.target_train()
        critic.target_train()
    return loss
