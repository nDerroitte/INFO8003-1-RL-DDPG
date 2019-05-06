from catcher import *
from gui import *
from ddpg import *
from Actor import *
from ReplayBuffer import *
from Critic import *
from OU import *
import time
import h5py
import tensorflow as tf

def load_weights(actor, critic):
        try:
            actor.model.load_weights('actor_w.h5')
        except:
            print("No 'actor_w.h5' file found, using new weights.")
        try:
            critic.model.load_weights('critic_w.h5')
        except:
            print("No 'critic_w.h5' file found, using new weights.")

def save_weights(actor, critic):
        actor.model.save_weights('actor_w.h5')
        critic.model.save_weights('critic_w.h5')
        #print("Actor and Critic weights saved.")

def endPrint(count, a):
    if a[0] == 1.0 or a[0] == -1.0:
        return count + 1
    return count


if __name__ == "__main__":
    game = ContinuousCatcher(dt=30)
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
    nb_episode = 300
    epsilon = 0.5
    EPS = 0.5
    load_weights(actor, critic)

    for episode_nb in range(nb_episode):
        print("=================================================================")
        print("Running simulation {}. Value of Epsilon : {}".format(episode_nb, EPS))
        d = False
        n_iterations = 0
        count = 0
        while d == False:
            states =  game.observe()
            act, tmp = playGame(sess, states, actor, 1, epsilon, action_dim, EPS)
            count = endPrint(count, tmp)
            n_iterations += 1
            next_states, r, d, r2 = game.step(act[0])
            loss = trainModels(buff, sess, states, r, next_states, act[0], loss, actor, critic, 1, episode_nb+1, epsilon)
            save_weights(actor, critic)
            gui.updateGUI(game, episode_nb)
        #buff.erase()
        print("Number of fruit catch : {}".format(game.nb_fruit_catch))
        print("Pourcentage of -1/1 : {}".format((count/n_iterations)*100))
        EPS = epsilon - (episode_nb/nb_episode)*epsilon
        game.reset()
    gui.makeVideo("Catcher")
    gui.closeGUI()
