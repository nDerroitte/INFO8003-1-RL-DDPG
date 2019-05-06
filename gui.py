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
        self.font = pygame.font.SysFont("monospace", 20)
        self.window_size = (width, height)
        self.window = pygame.display.set_mode(self.window_size)
        self.run_time = pygame.time.Clock()
        self.stream_img = []
        self.dt = dt

    def updateGUI(self, environemnt, nb_episode):
        self.__draw(environemnt, nb_episode)
        self.run_time.tick(self.dt)

    def __draw(self, environemnt, nb_episode):
        self.window.fill(CST.SCREEN_COLOR)
        bar_position = list(environemnt.bar.center)
        bar_position[0] -= round(self.bar_size[0]/2)
        fruit_position = environemnt.fruit.center
        fruit_position -= round(self.fruit_size[0]/2)
        pygame.draw.rect(self.window , CST.BAR_COLOR, (bar_position, self.bar_size))
        pygame.draw.rect(self.window , CST.FRUIT_COLOR, (fruit_position, self.fruit_size))
        str1 = "Episode : {}".format(nb_episode)
        text1 = self.font.render(str1, True, (255, 255, 0))
        self.window.blit(text1,(15,15))
        str2 = "Number of fruits caught : {}".format(environemnt.nb_fruit_catch)
        text2 = self.font.render(str2, True, (255, 255, 0))
        self.window.blit(text2,(15,40))
        str3 = "Number of life(s) remeaning: {}".format(environemnt.lives)
        text3 = self.font.render(str3, True, (255, 255, 0))
        self.window.blit(text3,(15,65))
        pygame.display.flip()
        # Video settings
        #img = pygame.surfarray.array3d(pygame.display.get_surface())
        #img = img.swapaxes(0, 1)
        #self.stream_img.append(img)


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
