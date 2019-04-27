from catcher import *
from gui import *
from ddpg import *
from Actor import *
from ReplayBuffer import *
from Critic import *
from OU import *
import time
import tensorflow as tf


if __name__ == "__main__":
    game = ContinuousCatcher()
    d = False
    gui = GUI(game.width, game.height, game.bar_width, game.bar_height,
              game.fruit_size, game.dt)

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    loss = 0


    for episode_nb in range(100):
        print("Running simulation {}.".format(episode_nb))
        d = False
        while d == False:
            states =  game.observe()
            act = playGame(sess, states, actor, 1)
            next_states, r, d = game.step(act)
            loss = trainModels(buff, sess, states, r ,next_states, act, loss,actor, critic, 1)
            gui.updateGUI(game, episode_nb)
        game.reset()
    gui.makeVideo("VideoName")
    gui.closeGUI()
