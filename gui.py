import pygame
from constants import CST

class GUI:
    def __init__(self, width, height, bar_width, bar_height, fruit_size):
        self.width = width
        self.height = height
        self.bar_size = (bar_width, bar_height)
        self.fruit_size = (fruit_size, fruit_size)
        pygame.init()
        self.window_size = (width, height)
        self.window = pygame.display.set_mode(self.window_size)
        self.run_time = pygame.time.Clock()

    def updateGUI(self, environemnt):
        self.__draw(environemnt)
        self.run_time.tick(environemnt.dt)

    def __draw(self, environemnt):
        self.window.fill(CST.SCREEN_COLOR)
        bar_position = environemnt.bar.center
        fruit_position = environemnt.fruit.center
        pygame.draw.rect(self.window , CST.BAR_COLOR, (bar_position, self.bar_size))
        pygame.draw.rect(self.window , CST.FRUIT_COLOR, (fruit_position, self.fruit_size))
        pygame.display.flip()


    def closeGUI(self):
        pygame.display.quit()
        pygame.quit()
