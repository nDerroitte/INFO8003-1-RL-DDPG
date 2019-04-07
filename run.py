from catcher import *
import time

if __name__ == "__main__":
    game = ContinuousCatcher()
    d = False
    while d == False:
        act = game.observe()
        a, r, d = game.step(act)
        print(r, d)
