#!/usr/bin/python3.6

import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from Simulation import Simulation


IMPLEMENTED_POLICY = ["Epsilon-Greedy", "Actor-Critic"]

IMPLEMENTED_ALGORITHMS = ["Fitted-Q Learning", "Q-Learning"]

IMPLEMENTED_FITTED_Q_MODEL = ["Linear Regression", "Extremely Randomized Trees", "Neural Networks"]

IMPLEMENTED_Q_LEARNING_APPROXIMATOR_MODEL = ['Neural Networks', 'Radial Basis']

ACTOR_MODEL = ["Neural Network 1"]

CRITIC_MODEL = ["Neural Network 1"]


def main(args):
    nb_figure = 0

    simu = Simulation(gamma=args.gamma)

    final_rewards = None

    if args.policy == IMPLEMENTED_POLICY[0]:
        final_rewards, test_rewards = simu.run_greedy(args.nb_episode, args.episode_size, args.N, args.epsilon,
                                           model_name=args.model_name,
                                           algorithm=args.algorithm,
                                           exploration_decay=args.exploration_decay,
                                           epsilon_decay=args.epsilon_decay,
                                           experience_replay=args.experience_replay,
                                           alpha=args.alpha,
                                           load_model=args.load_model,
                                           make_video=args.make_video,
                                           store_model=args.store_model,
                                           run = args.run,
                                           display=(not args.no_display)
                                           )

    elif args.policy == IMPLEMENTED_POLICY[1]:
        final_rewards, test_rewards = simu.run_actor_critic(args.nb_episode, args.episode_size,
                                              actor_model_name=args.actor_model_name,
                                              critic_model_name=args.critic_model_name,
                                              epsilon=args.epsilon,
                                              exploration_decay=args.exploration_decay,
                                              epsilon_decay=args.epsilon_decay,
                                              load_model=args.load_model,
                                              make_video=args.make_video,
                                              store_model=args.store_model,
                                              run=args.run,
                                              display=(not args.no_display)
                                               )

    else:
        print(("Error: Unimplemented " + args.policy + " policy."),
              file=sys.stderr)

        exit(-1)

    img = plt.figure("Reward Evolution while Training")

    plt.title("Training Reward Evolution")
    plt.xlabel("Episodes")
    plt.ylabel("Episode's total reward")

    plt.plot(range(0, len(final_rewards)), final_rewards)

    plt.savefig(("reward_evol_" + str(args.policy)), bbox_inches='tight', dpi=100)

    plt.show()

    input()

    plt.close()

    img2 = plt.figure("Test Reward Evolution")

    plt.title("Test Reward Evolution")
    plt.xlabel("Episodes")
    plt.ylabel("Episode's total reward")

    plt.plot(range(0, len(test_rewards) * 50, 50), test_rewards)

    plt.savefig(("test_reward_evol_" + str(args.policy)), bbox_inches='tight', dpi=100)

    plt.show()

    input()

    plt.close()


if __name__ == "__main__":
    usage = """
    USAGE:      python3 main.py <options>
    EXAMPLES:   (1) python main.py  --nb_episode 100 --episode_size 100 --policy "Epsilon-Greedy" --epsilon 0.5 
                        --model_name "Linear Regression"
                        
                    - Launch the Simulation with an Epsilon-Greedy policy using a Linear Regression Model
    """

    parser = ArgumentParser(usage)

    parser.add_argument(

        '--nb_episode',
        help="""
                The number of episode of the simulation that you would like to run.
        """,
        type=int,
        default=100,
        required=True,
    )

    parser.add_argument(

        '--episode_size',
        help="""
                The size of the episode of the simulation that you would like to run.
        """,
        type=int,
        default=100,
        required=True,
    )

    parser.add_argument(

        '--N',
        help="""
                The Number of iteration of the Fitted Q-Iteration algorithm or of the Parametric Q-Learning that 
                                you would like to run during the simulation.
        """,
        type=int,
        default=50,
    )

    parser.add_argument(

        '--policy',
        help="""
                The Policy that you would like to use during the simulation.
        """,
        type=str,
        default=None,
        required=True,
        choices=IMPLEMENTED_POLICY
    )

    parser.add_argument(

        '--algorithm',
        help="""
                The Algorithm that you would like to use during the simulation.
        """,
        type=str,
        default=None,
        required="Actor-Critic" not in sys.argv,
        choices=IMPLEMENTED_ALGORITHMS
    )

    parser.add_argument(

        '--epsilon',
        help="""
                The epsilon parameter of the Epsilon-Greedy policy that will be run during the simulation.
        """,
        type=float,
        default=None,
        required="Epsilon-Greedy" in sys.argv or "Actor-Critic" in sys.argv,
    )

    parser.add_argument(

        '--exploration_decay',
        help="""
                The exploration decay of the Epsilon-greedy policy that will be run during the simulation.
                    (how the epsilon decrease with the number of episodes that has already been run)
        """,
        dest='exploration_decay',
        action='store_true',
    )

    parser.add_argument(

        '--epsilon_decay',
        help="""
                The exploration decay of the Epsilon-greedy policy that will be run during the simulation.
                    (how the epsilon decrease with the number of episodes that has already been run)
        """,
        type=float,
        default=None
    )

    parser.add_argument(

        '--experience_replay',
        help="""
                Whether or not to use experience replay during the simulation.
        """,
        dest='experience_replay',
        action='store_true',
    )

    if "Q-Learning" in sys.argv:
        models = IMPLEMENTED_Q_LEARNING_APPROXIMATOR_MODEL
    else:
        models = IMPLEMENTED_FITTED_Q_MODEL

    parser.add_argument(
        '--model_name',
        help="""
                The model that will be used for the Fitted Q-Iteration or for the Paramteric Q-Learning algorithm
                                                    of the simulation.
        """,
        type=str,
        required="Actor-Critic" not in sys.argv,
        choices=models
    )

    parser.add_argument(
        '--actor_model_name',
        help="""
                        The model that will be used for the Actor or for the Actor-Critic algorithm
                                                    of the simulation.
        """,
        type=str,
        required="Actor-Critic"in sys.argv,
        choices=ACTOR_MODEL
    )

    parser.add_argument(
        '--critic_model_name',
        help="""
                        The model that will be used for the Critic or for the Actor-Critic algorithm
                                                    of the simulation.
        """,
        type=str,
        required="Actor-Critic"in sys.argv,
        choices=CRITIC_MODEL
    )

    parser.add_argument(
        '--no_display',
        help="""
                            Wether to display or not the GUI while training.
             """,
        dest='no_display',
        action='store_true',
    )

    parser.add_argument(
        '--gamma',
        help="""
                The value of the discount factor which is a number between 0 and 1 and has the effect of valuing rewards 
                                received earlier higher than those received later.
                γ may also be interpreted as the probability to succeed (or survive) at every step Δt. 
        """,
        type=float,
        default=0.95
    )

    parser.add_argument(
        '--initial_state',
        help="""
                The initial state (p, s) of the agent when evolving on the domain
                    where 'p' is the initial position and 's' is the initial speed.
        """,
        type=float,
        nargs=2,
        default=[0, 0]
    )

    parser.add_argument(
        '--no_graphs',
        help="""
                Wether or not graphs should be displayed during the simulation.
            """,
        dest='no_graphs',
        action='store_true',
    )

    parser.add_argument(

        '--alpha',
        help="""
                The alpha parameter of the Parametric Q-Learning Algorithm that will be run during the simulation.
        """,
        type=float,
        default=None,
        required="Q-Learning" in sys.argv,
    )

    parser.add_argument(
        "--load_model",
        help="""
                Wether or not models should be loaded from files.
            """,
        dest='load_model',
        action='store_true',

    )

    parser.add_argument(
        "--make_video",
        help="""
                Wether or not video should be done of the running (WARNING : memory greedy).
            """,
        dest='make_video',
        action='store_true',

    )

    parser.add_argument(
        "--store_model",
        help="""
                Wether or not models should be stored.
            """,
        dest='store_model',
        action='store_true',

    )

    parser.add_argument(
        "--run",
        help="""
                Only run the simulation without any training;
            """,
        dest='run',
        action='store_true',

    )

    parser.set_defaults(no_graphs=False)
    parser.set_defaults(no_display=False)
    parser.set_defaults(q_learning=False)
    parser.set_defaults(experience_replay=False)
    parser.set_defaults(exploration_decay=False)
    parser.set_defaults(load_model=False)
    parser.set_defaults(make_video=False)
    parser.set_defaults(store_model=False)
    parser.set_defaults(run=False)

    args = parser.parse_args()

    main(args)
