import pygame
import cv2
from constants import CST

class GUI:
    def __init__(self, width, height, bar_width, bar_height, fruit_size, dt):
        self.width = width
        self.height = height
        self.bar_size = (bar_width, bar_height)
        self.fruit_size = (fruit_size, fruit_size)
        pygame.init()
        self.window_size = (width, height)
        self.window = pygame.display.set_mode(self.window_size)
        self.run_time = pygame.time.Clock()
        self.stream_img = []
        self.dt = dt

    def updateGUI(self, environemnt):
        self.__draw(environemnt)
        self.run_time.tick(self.dt)

    def __draw(self, environemnt):
        self.window.fill(CST.SCREEN_COLOR)
        bar_position = environemnt.bar.center
        fruit_position = environemnt.fruit.center
        pygame.draw.rect(self.window , CST.BAR_COLOR, (bar_position, self.bar_size))
        pygame.draw.rect(self.window , CST.FRUIT_COLOR, (fruit_position, self.fruit_size))
        pygame.display.flip()
        # Video settings
        img = pygame.surfarray.array3d(pygame.display.get_surface())
        img = img.swapaxes(0, 1)
        self.stream_img.append(img)


    def closeGUI(self):
        pygame.display.quit()
        pygame.quit()

    def makeVideo(self, name):
        print("Creating Video ..", end=' ')
        # Get video stream dimensions from the image size
        if len(self.stream_img) > 0:
            height, width, layer = self.stream_img[0].shape
            size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter(name + '.avi', fourcc, self.dt, size)
            for i in range(len(self.stream_img)):
                out.write(cv2.cvtColor(self.stream_img[i], cv2.COLOR_RGB2BGR))
            out.release()
            print("done!")
            cv2.destroyAllWindows()
