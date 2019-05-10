from argparse import ArgumentParser, ArgumentTypeError
from catcher import *
from gui import *
from ddpg import *
from actor import *
from replay import *
from critic import *
from OU import *
from plot import *
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
        type=float,
        default=0.0
    )

    # Plot
    parser.add_argument(
        '--plot',
        help="""1 if we want the plot. 0 otherwise""",
        type=int,
        default = 0
    )

    # Video
    parser.add_argument(
        '--video',
        help="""1 if we want to make a video. 0 otherwise""",
        type=int,
        default = 0
    )
    # Arguments parsing
    args = parser.parse_args()
    nb_episode = args.nb_episodes
    train = args.train
    video = args.video
    plot = args.plot
    max_epsilon = args.epsilon

    # Initiate variables
    d = False
    loss = 0
    current_epsilon = max_epsilon
    max_force = 0.021 * 640

    # Initiate game related objects
    game = ContinuousCatcher(dt=30)
    gui = GUI(game.width, game.height, game.bar_width, game.bar_height,
              game.fruit_size, game.dt, video)

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

    if plot:
        plot_object = Plot()

    # Looping on each episode
    for episode_nb in range(nb_episode):
        # Plot making
        if(episode_nb%10 == 0 and plot):
            plot_object.prepare_plot(sess, actor, action_dim, game, gui,
                                     episode_nb)

        print("==============================================================")
        print("Running simulation {}.".format(episode_nb)+
                 "Value of epsilon : {}".format(current_epsilon))
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
                                   loss, actor, critic)
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
    if video:
        gui.makeVideo("Catcher")
    if plot:
        plot_object.make_plot()
    gui.closeGUI()
