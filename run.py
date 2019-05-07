from argparse import ArgumentParser, ArgumentTypeError
from catcher import *
from gui import *
from ddpg import *
from actor import *
from replay import *
from critic import *
from OU import *
import time
import h5py
import tensorflow as tf
from keras import backend as K

if __name__ == "__main__":
    usage = """
    USAGE:      python3 run.py <options>
    EXAMPLES:   (1) python run.py
                    - Launch the Q4 of the project : Visualisation of a policy
    """

    # Using argparse to select the different setting for the run
    parser = ArgumentParser(usage)
    # nb_episodes
    parser.add_argument(
        '--nb_episodes',
        help='Number of episodes used for the DDPG',
        type=int,
        default=1
    )

    # Train indicator
    parser.add_argument(
        '--train',
        help='Enable the train mode',
        type=int,
        default=0
    )

    # Epsilon
    parser.add_argument(
        '--epsilon',
        help="""Value of the epsilon parameter
                (regarding the exploration/exploitation dilemma)""",
        type=int,
        default=0
    )
    # Arguments parsing
    args = parser.parse_args()
    nb_episode = args.nb_episodes
    train = args.train
    max_epsilon = args.epsilon

    # Initiate variables
    d = False
    loss = 0
    current_epsilon = max_epsilon
    max_force = 0.021 * 640

    # Initiate game related objects
    game = ContinuousCatcher(dt=30)
    gui = GUI(game.width, game.height, game.bar_width, game.bar_height,
              game.fruit_size, game.dt)

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # Initiate networks and replay buffer
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU,
                         max_force)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)

    load_weights(actor, critic)

    # Looping on each episode
    for episode_nb in range(nb_episode):
        print("==============================================================")
        print("Running simulation {}."+
                 "Value of epsilon : {}".format(episode_nb, current_epsilon))
        # (Re)-initialise loop related variable
        d = False
        total_reward = 0
        while d == False:
            # Get observations
            states =  game.observe()
            # Generate the action
            act = getAction(sess, states, actor, train, max_epsilon,
                                action_dim, current_epsilon)
            # Perform the action2
            next_states, r, d = game.step(act[0])
            total_reward += r
            # Train the model if necessery
            if train:
                loss = trainModels(buff, sess, states, r, next_states, act[0],
                                   loss, actor, critic, episode_nb+1,
                                   max_epsilon)
                save_weights(actor, critic)
            # Update GUI
            gui.updateGUI(game, episode_nb)
        # End loop print
        print("")
        print("Number of fruit catch : {}".format(game.nb_fruit_catch))
        print("Total reward: {}".format(total_reward))
        # Saving weights under specifi name at each end of episode
        if train:
            print("Saving weights!")
            save_weights(actor, critic, True)
        # Update epsilon
        current_epsilon = max_epsilon - (episode_nb/nb_episode)*max_epsilon
        # Reset the game
        game.reset()
    # Create a video sequence and close GUI
    gui.makeVideo("Catcher")
    gui.closeGUI()
