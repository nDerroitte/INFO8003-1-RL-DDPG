#!/usr/bin/python3.6

import sys
from argparse import ArgumentParser

from Simulation import Simulation
from Util import *

QUESTIONS = ["2", "3", "5", "6", "all"]
IMPLEMENTENTED_POLICY = ["Epsilon-Greedy", "Softmax"]
IMPLEMENTENTED_MODEL = ["Linear Regression", "Extremely Randomized Trees", "Neural Networks"]
IMPLEMENTENTED_Q_LEARNING_APPROXIMATOR_MODEL = ['Neural Networks', 'Radial Basis']


def main(args):
    nb_figure = 0

    simu = Simulation(gamma=args.gamma)

    if args.question == "all":
        display_graphs = False
    else:
        display_graphs = not args.no_graphs

    if args.question is None:
        final_rewards = None

        if args.policy == IMPLEMENTENTED_POLICY[0]:
            final_rewards, q = simu.run_greedy(args.nb_episode, args.episode_size, args.N, args.epsilon,
                                               model_name=args.model_name,
                                               exploration_decay=args.exploration_decay,
                                               experience_replay=args.experience_replay,
                                               alpha=args.alpha,
                                               display=True
                                               )

        elif args.policy == IMPLEMENTENTED_POLICY[1]:
            final_rewards, q = simu.run_softmax(args.nb_episode, args.episode_size, args.N,
                                                args.temperature,
                                                model_name=args.model_name,
                                                experience_replay=args.experience_replay,
                                                alpha=args.alpha,
                                                display=True
                                                )

        else:
            print(("Error: Unimplemented " + args.policy + " policy."),
                  file=sys.stderr)
            exit(-1)


if __name__ == "__main__":
    usage = """
    USAGE:      python3 main.py <options>
    EXAMPLES:   (1) python main.py  --nb_episode 100 --episode_size 100 --policy "Epsilon-Greedy" --epsilon 0.5 
                        --model_name "Linear Regression"
                        
                    - Launch the "Car on the Hill" Simulation with an Epsilon-Greedy policy using 
                                            a Linear Regression Model
                                            
                (2) python main.py --question 2
                
                    - Launch the Question 2 of the Project 2's statement.
    """

    # Using argparse to select the different setting for the run
    parser = ArgumentParser(usage)

    parser.add_argument(
        '--question',
        help="""
                The question of the project that you would like to run.
            (cfr. statement of Project 2 of Optimal Decision Making course)
            
        """,
        type=str,
        default=None,
        choices=QUESTIONS
    )

    parser.add_argument(

        '--nb_episode',
        help="""
                The number of episode of the simulation that you would like to run.
        """,
        type=int,
        default=100,
        required="--question" not in sys.argv,
    )

    parser.add_argument(

        '--episode_size',
        help="""
                The size of the episode of the simulation that you would like to run.
        """,
        type=int,
        default=100,
        required="--question" not in sys.argv,
    )

    parser.add_argument(

        '--N',
        help="""
                The Number of iteration of the Fitted Q-Iteration algorithm or of the Parametric Q-Learning that 
                                you would like to run during the simulation.
        """,
        type=int,
        default=100,
    )

    parser.add_argument(

        '--policy',
        help="""
                The Policy that you would like to use during the simulation.
        """,
        type=str,
        default=None,
        required="--question" not in sys.argv,
        choices=IMPLEMENTENTED_POLICY
    )

    parser.add_argument(

        '--epsilon',
        help="""
                The epsilon parameter of the Epsilon-Greedy policy that will be run during the simulation.
        """,
        type=float,
        default=None,
        required="Epsilon-Greedy" in sys.argv,
    )

    parser.add_argument(

        '--exploration_decay',
        help="""
                The exploration decay of the Epsilon-greedy policy that will be run during the simulation.
                    (how the epsilon decrease with the number of episodes that has already been run)
        """,
        type=float,
        default=None,
    )

    parser.add_argument(

        '--temperature',
        help="""
                The temperature parameter of the Softmax policy that will be run during the simulation.
        """,
        type=float,
        default=None,
        required="Softmax" in sys.argv,
    )

    parser.add_argument(

        '--experience_replay',
        help="""
                Wether or not to use experience replay during the simulation.
        """,
        type=bool,
        default=False,
    )

    if "--q_learning" in sys.argv:
        models = IMPLEMENTENTED_Q_LEARNING_APPROXIMATOR_MODEL
    else:
        models = IMPLEMENTENTED_MODEL

    parser.add_argument(
        '--model_name',
        help="""
                The model that will be used for the Fitted Q-Iteration or for the Paramteric Q-Learning algorithm
                                                    of the simulation.
        """,
        type=str,
        required="--question" not in sys.argv,
        choices=models
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

        '--q_learning',
        help="""
                Wether or not Parametric Q-Learning Algorithm should be used during the simulation.
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
        required="--q_learning" in sys.argv,
    )

    parser.set_defaults(no_graphs=False)
    parser.set_defaults(q_learning=False)

    args = parser.parse_args()

    main(args)
