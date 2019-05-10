from ActorCritic import *
from Fitted_Q_Actor import *
from Q_Learner_Actor import *

from Util import *
from catcher import ContinuousCatcher
from gui import *


class Simulation:
    def __init__(self, gamma):
        self.agent = None

        self.gamma = gamma

        self.already_fitted = False

    """
        Value Based Methods        
    """

    def run_greedy(self, nb_episode, episode_size, N, epsilon, model_name, algorithm, epsilon_decay=None,
                   exploration_decay=False, experience_replay=False, alpha=None, load_model=False, make_video=False,
                   store_model=False, run=False, display=False):

        self.agent = ContinuousCatcher(init_lives=episode_size)

        if algorithm == "Fitted-Q Learning":
            actor = Fitted_Q_Actor(self.agent, epsilon, model_name, self.gamma, N, epsilon_decay, load_model)

        elif algorithm == "Q-Learning":
            actor = Q_Learner_Actor(self.agent, epsilon, model_name, self.gamma, N, alpha, epsilon_decay, load_model)

        else:
            raise Exception("Error : Unkwown algorithm " + str(algorithm))

        print("Running Simulation with an Epsilon-greedy policy and a " + str(algorithm) + " algorithm using a "
              + model_name + " model(" + 'exploration_decay=' + str(exploration_decay) + ', experience_replay='
              + str(experience_replay) + "):")

        if exploration_decay is True and epsilon_decay is None:
            epsilon_decay = epsilon / nb_episode

        e = epsilon

        final_rewards = list()

        if display is True:
            gui = GUI(make_video=make_video)

        for i in range(1, nb_episode + 1):

            if display is True:
                gui.updateGUI(self.agent, i)

            if algorithm == "Fitted-Q Learning":
                print("Episode ", i, " : Fitted-Q Learning computation...")

            elif algorithm == "Q-Learning":
                print("Episode ", i, " : Parametric Q Learning computation...")

            else:
                raise Exception("Error: Unimplemented algorithm " + str(algorithm))
            done = False
            while done is False:
                state, reward, done = self.update_greedy(actor)

                if display is True:
                    gui.updateGUI(self.agent, i)

            total_reward = self.agent.get_total_reward()
            final_rewards.append(total_reward)

            print("")
            print("Episode ", str(i), " : Total Reward = ", str(self.agent.get_total_reward()))

            if exploration_decay is True:
                e = e - epsilon_decay
                actor.set_epsilon(e)
            agent_history = self.agent.get_history()

            if experience_replay is True:
                agent_history = self.apply_experience_replay(agent_history)

            h.append(agent_history)

            self.agent.reset()

            if display is True:
                gui.updateGUI(self.agent, i)

            total_reward = self.agent.get_total_reward()
            final_rewards.append(total_reward)

            if store_model is True and i % 50 == 0:
                print("")
                done = False
                while done is False:
                    state, reward, done = self.update_actor_critic(actor_critic, run=False)
                    if display is True:
                        gui.updateGUI(self.agent, i)

                test_reward = self.agent.get_total_reward()

                print("Model Testing Results : Total Reward =", str(test_reward), ".")

                test_rewards.append(test_reward)

                self.agent.reset()
                actor_critic.store_models()

            if store_model is True and i % 50 == 0:
                print("")
                done = False
                while done is False:
                    state, reward, done = self.update_actor_critic(actor_critic, run=False)
                    if display is True:
                        gui.updateGUI(self.agent, i)

                test_reward = self.agent.get_total_reward()

                print("Model Testing Results : Total Reward =", str(test_reward), ".")

                test_rewards.append(test_reward)

                self.agent.reset()
                actor_critic.store_models()

        if display is True and make_video is True:
            gui.makeVideo("Simulation")

        if display is True:
            gui.closeGUI()

        return final_rewards

    def update_greedy(self, actor):
        curr_state = self.agent.observe()

        move = actor.act(curr_state)

        next_state, reward, done = self.agent.step(move)

        return next_state, reward, done

    """
        Actor-Critics Based Methods
    """

    def run_actor_critic(self, nb_episode, episode_size, actor_model_name, critic_model_name, epsilon, epsilon_decay,
                         exploration_decay=False, load_model=False, make_video=False, store_model=False,
                         run=False, display=False):

        self.agent = ContinuousCatcher(init_lives=episode_size)

        h = list()
        final_rewards = list()
        test_rewards = list()

        if load_model is True:
            with open("final_rewards.txt", 'r') as f:
                final_rewards.append(f.readline())

            with open("test_rewards.txt", 'r') as f:
                test_rewards.append(f.readline())

        if display is True:
            gui = GUI(make_video=make_video)

        if exploration_decay is True and epsilon_decay is None:
            epsilon_decay = epsilon / nb_episode

        e = epsilon

        actor_critic = ActorCritic(self.agent, epsilon, actor_model_name, critic_model_name, self.gamma,
                                   epsilon_decay, load_model)

        self.already_fitted = False

        print("Running Simulation with an Actor-critic algorithm:")

        for i in range(1, nb_episode + 1):
            done = False
            while done is False:
                state, reward, done = self.update_actor_critic(actor_critic, run)
                if display is True:
                    gui.updateGUI(self.agent, i)

            print("Episode ", str(i), " : Total Reward = ", str(self.agent.get_total_reward()), " - Epsilon =", e,
                  end="\r")

            if exploration_decay is True:
                e = e - epsilon_decay
                actor_critic.set_epsilon(e)

            total_reward = self.agent.get_total_reward()
            final_rewards.append(total_reward)

            agent_history = self.agent.get_history()

            h.append(agent_history)

            self.agent.reset()

            if display is True:
                gui.updateGUI(self.agent, i)

            if store_model is True and (i % 50 == 0 or i == nb_episode):
                print("")

                """
                        TESTING THE MODEL
                """

                actor_critic.set_epsilon(0)

                done = False
                while done is False:
                    state, reward, done = self.update_actor_critic(actor_critic, run=False)
                    if display is True:
                        gui.updateGUI(self.agent, i)

                test_reward = self.agent.get_total_reward()

                actor_critic.set_epsilon(e)

                print("Model Testing Results : Total Reward =", str(test_reward), ".")

                test_rewards.append(test_reward)

                self.agent.reset()
                actor_critic.store_models()

                with open("final_rewards.txt", 'w') as f:
                    for s in final_rewards:
                        f.write(str(s) + '\n')

                with open("test_rewards.txt", 'w') as f:
                    for s in test_rewards:
                        f.write(str(s) + '\n')

        if display is True and make_video is True:
            print("")
            gui.makeVideo("Simulation")

        if display is True:
            gui.closeGUI()

        return final_rewards, test_rewards

    def update_actor_critic(self, actor_critic, run):
        curr_state = self.agent.observe()

        move = actor_critic.act(curr_state.reshape(1, -1))

        next_state, reward, done = self.agent.step(move)

        if run is not True:
            actor_critic.train(curr_state.reshape(1, -1), move, reward, next_state.reshape(1, -1), done)

        return next_state, reward, done
