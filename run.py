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
    # gui = GUI(game.width, game.height, game.bar_width, game.bar_height,
            #   game.fruit_size, game.dt)

    #Tensorflow GPU optimization
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    
    # K.set_session(sess)
    sess = 0
    # critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    loss = 0

    action_list = list(range(-10,10))
    # action_list = [-4,0, 4]
    training_set = np.array([[]])
    rewards_list = np.array([])
    next_states_set = np.array([[]])
    actions = np.array([])

    # act = action_list[0]
    # next_states, r, d = game.step([act])
    # training_set = np.concatenate((training_set,[np.append(state,[act])]),axis=1)
    # rewards_list = np.append(rewards_list,r)
    # next_states_set = np.concatenate((next_states_set,[np.append(next_states,[act])]),axis=1)
    # actions = np.append(actions,act)

    # act = action_list[1]
    # next_states, r, d = game.step([act])
    # training_set = np.concatenate((training_set,[np.append(state,[act])]))
    # rewards_list = np.append(rewards_list,r)
    # print(rewards_list)
    # next_states_set = np.concatenate((next_states_set,[np.append(next_states,[act])]))
    # actions = np.append(actions,act)

    
    while d == False:
        state =  game.observe()
        act = action_list[random.randint(0,6)]
        next_states, r, d = game.step([act])
        # loss = trainModels(buff, sess, states, r ,next_states, act, loss,actor, critic, 1)
        # gui.updateGUI(game, -1)
        # print(rewards_list)
        buff.add(state,act,r,next_states)
    print("nb fruits caught",game.nb_fruit_catch)
    game.reset()
    batch = buff.getBatch(BATCH_SIZE)
    states = np.asarray([e[0] for e in batch])
    actions = np.asarray([e[1] for e in batch])
    rewards = np.asarray([e[2] for e in batch])
    new_states = np.asarray([e[3] for e in batch])
    
    # states = np.array(states)
    # actions = np.array([actions])
    # states = np.concatenate((states,actions.T),axis=1)

    actor = ActorNetwork(sess, state_dim, action_dim, GAMMA, TAU, states, actions, rewards)
    # print("rewards : ",rewards)
    # time.sleep(120)
    # actor.train(training_set,rewards_list,next_states_set)
    # actor.train_batch(training_set, actions, rewards_list, next_states_set,action_list)
    for i in range(20):
        actor.train_batch(states, actions, rewards, new_states,action_list)
    for episode_nb in range(1500):
        print("Running simulation {}.".format(episode_nb))
        d = False
        while d == False:
            state =  game.observe()
            act = actor.predict(state,action_list)
            next_states, r, d = game.step([act])
            # gui.updateGUI(game, episode_nb)
            buff.add(state,act,r,next_states)
        print("nb fruits caught",game.nb_fruit_catch)
        game.reset()
        batch = buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])

        # states = np.array(states)
        # actions = np.array([actions])
        # states = np.concatenate((states,actions.T),axis=1)

        for i in range(20):
            actor.train_batch(states, actions, rewards, new_states,action_list)
    # gui.makeVideo("VideoName")
    # gui.closeGUI()
