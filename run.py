from catcher import *
from gui import *
import time

if __name__ == "__main__":
    game = ContinuousCatcher()
    d = False
    gui = GUI(game.width, game.height, game.bar_width, game.bar_height,
              game.fruit_size, game.dt)
    gui.updateGUI(game)
    while d == False:
        act = game.observe()
        a, r, d = game.step(act)
        gui.updateGUI(game)
    gui.makeVideo("VideoName")
    gui.closeGUI()
