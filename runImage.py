import numpy as np
from constants import CST


def e_greedy(epsilon, x, u):
    e = np.random.rand()
    if e < epsilon:
        # TO DO CHANGEEEEE
        move = np.random.uniform[-5,5]
    else:
        move = u
    return move


if __name__ == "__main__":
    game = ContinuousCatcher()
    d = False
    gui = GUI(game.width, game.height, game.bar_width, game.bar_height,
              game.fruit_size, game.dt)
    model = None
    for episode_nb in range(5):
        print("Running simulation {}.".format(episode_nb))
        d = False
        while d == False:
            act = game.observe()
            a, r, d = game.step(act)
            gui.updateGUI(game, episode_nb)
        X_img = gui.stream_img
        gui.stream_img = []
        history = game.get_history
        for i in range(len(X_img)):
            if X_move[i] is not None:
                last_move = X_move[i]
            else:
                last_move = np.random.uniform[-5,5]
                X_move.append(0)
            X_move[i] = e_greedy(0.25, X_img[i], last_move)

        if model is None:
            history = np.asarray(history)
            Y = history[:,4]
        else:
            for i in range(len(X)):
                Y.append(history[:,4] + CST.GAMMA * model.predict(X[i]))
        X = np.hstack([X_img, X_move])
        #model = CNN
        # update policy
