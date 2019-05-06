from catcher import *
from gui import *
from ddpg import *
from Actor import *
from ReplayBuffer import *
from Critic import *
from OU import *
import time
import numpy as np


if __name__ == "__main__":
    game = ContinuousCatcher()
    d = False
    gui = GUI(game.width, game.height, game.bar_width, game.bar_height,
              game.fruit_size, game.dt)

    #Tensorflow GPU optimization
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    
    # K.set_session(sess)
    sess = 0
    # critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    loss = 0

    action_list = [-5, -2, 0, 2, 5]
    training_set = np.array([[]])
    rewards_list = np.array([])
    next_states_set = np.array([[]])

    state = game.observe()

    act = action_list[1]
    next_states, r, d = game.step([act])
    training_set = np.concatenate((training_set,[np.append(state,[act])]),axis=1)
    rewards_list = np.append(rewards_list,r)
    next_states_set = np.concatenate((next_states_set,[np.append(next_states,[act])]),axis=1)

    act = action_list[3]
    next_states, r, d = game.step([act])
    training_set = np.concatenate((training_set,[np.append(state,[act])]))
    rewards_list = np.append(rewards_list,r)
    next_states_set = np.concatenate((next_states_set,[np.append(next_states,[act])]))
    
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, training_set, rewards_list)

    for episode_nb in range(500):
        print("Running simulation {}.".format(episode_nb))
        d = False
        while d == False:
            state =  game.observe()
            act = actor.predict(state,action_list)
            next_states, r, d = game.step([act])
            # loss = trainModels(buff, sess, states, r ,next_states, act, loss,actor, critic, 1)
            gui.updateGUI(game, episode_nb)
            training_set = np.concatenate((training_set,[np.append(state,[act])]))
            next_states_set = np.concatenate((next_states_set,[np.append(next_states,[act])]))
            rewards_list = np.append(rewards_list,r)
        print("nb fruits caught",game.nb_fruit_catch)
        game.reset()
        actor.train(training_set,rewards_list,next_states_set)
    gui.makeVideo("VideoName")
    gui.closeGUI()
