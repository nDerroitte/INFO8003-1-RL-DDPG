import pygame
import cv2
from constants import CST

class GUI:
    def __init__(self, width, height, bar_width, bar_height, fruit_size, dt, v):
        """
        Parameters:
        -----------
        width : int
            Width of the window
        height : int
            height of the window
        bar_width : int
            Width of the bar
        bar_height : int
            Height of the bar
        fruit_size : int
            size of the fruit
        dt : int
            timestep
        """
        # Init of the window
        self.width = width
        self.height = height
        self.bar_size = (bar_width, bar_height)
        self.fruit_size = (fruit_size, fruit_size)
        # Init the pygame setting
        pygame.init()
        self.font = pygame.font.SysFont("monospace", 20)
        self.window_size = (width, height)
        self.window = pygame.display.set_mode(self.window_size)
        self.run_time = pygame.time.Clock()
        self.stream_img = []
        self.dt = dt
        self.video_indicator = v

    def updateGUI(self, environemnt, nb_episode):
        """
        Update the GUI

        Parameters:
        -----------
        environemnt : game object
        nb_episode : int
            Episode count
        """
        self.__draw(environemnt, nb_episode)
        self.run_time.tick(self.dt)

    def __draw(self, env, nb_episode):
        """
        Draw the window with the current environemnt

        Parameters:
        -----------
        env : game object
        nb_episode : int
            Episode count
        """
        # Pygame drawing
        self.window.fill(CST.SCREEN_COLOR)
        bar_position = list(env.bar.center)
        bar_position[0] -= round(self.bar_size[0]/2)
        fruit_position = env.fruit.center
        fruit_position -= round(self.fruit_size[0]/2)
        # Drawing the fruit and the bar
        pygame.draw.rect(self.window , CST.BAR_COLOR,
                         (bar_position, self.bar_size))
        pygame.draw.rect(self.window , CST.FRUIT_COLOR,
                         (fruit_position, self.fruit_size))
        # Text display
        str1 = "Episode : {}".format(nb_episode)
        text1 = self.font.render(str1, True, (255, 255, 0))
        self.window.blit(text1,(15,15))
        str2 = "Number of fruits caught : {}".format(env.nb_fruit_catch)
        text2 = self.font.render(str2, True, (255, 255, 0))
        self.window.blit(text2,(15,40))
        str3 = "Number of life(s) remeaning: {}".format(env.lives)
        text3 = self.font.render(str3, True, (255, 255, 0))
        self.window.blit(text3,(15,65))
        pygame.display.flip()
        # Video settings
        if self.video_indicator:
            img = pygame.surfarray.array3d(pygame.display.get_surface())
            img = img.swapaxes(0, 1)
            self.stream_img.append(img)


    def closeGUI(self):
        """
        Quit GUI
        """
        pygame.display.quit()
        pygame.quit()

    def makeVideo(self, name):
        """
        Create a video file using opencv2

        Parameters:
        -----------
        name : str
            Name of the video
        """
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
