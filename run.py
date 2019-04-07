from catcher import *
from gui import *
import time

if __name__ == "__main__":
    game = ContinuousCatcher()
    d = False
    gui = GUI(game.width, game.height, game.bar_width, game.bar_height,
              game.fruit_size, game.dt)
    for episode_nb in range(5):
        print("Running simulation {}.".format(episode_nb))
        d = False
        while d == False:
            act = game.observe()
            a, r, d = game.step(act)
            gui.updateGUI(game, episode_nb)
        game.reset()
    gui.makeVideo("VideoName")
    gui.closeGUI()
