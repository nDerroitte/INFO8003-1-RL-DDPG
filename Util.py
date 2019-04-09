import numpy as np
import math

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pygame


def get_rand_number(min_value, max_value):
    """
    This function gets a random number from a uniform distribution between
    the two input values [min_value, max_value] inclusively
    Args:
    - min_value (float)
    - max_value (float)
    Return:
    - Random number between this range (float)
    """
    range = max_value - min_value
    choice = np.random.uniform(0, 1)

    return min_value + range * choice


def init_screen(window_name, canvas_width, canvas_height):
    pygame.init()
    pygame.display.set_caption(window_name)
    screen = pygame.display.set_mode((canvas_width, canvas_height))

    return screen


def store_expected_return(expected_return, fig_label, title, x_label, y_label, z_label, display=False):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-3, 3, 100)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    a = len(Z)
    b = len(Z[0])
    i = 0
    while i < a:
        j = 0
        while j < b:
            Z[i][j] = expected_return(X[i][j], Y[i][j])
            j = j + 1
        i = i + 1

    store_graph(X, Y, Z, fig_label, title, x_label, y_label, z_label, display=display)


def show_expected_return(expected_return, fig_label, title, x_label, y_label, z_label):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-3, 3, 100)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    a = len(Z)
    b = len(Z[0])
    i = 0
    while i < a:
        j = 0
        while j < b:
            Z[i][j] = expected_return(X[i][j], Y[i][j])
            j = j + 1
        i = i + 1

    show_graph(X, Y, Z, fig_label, title, x_label, y_label, z_label)


def store_policy(policy, fig_label, title, x_label, y_label, z_label, display=False):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-3, 3, 100)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    a = len(Z)
    b = len(Z[0])
    i = 0

    while i < a:
        j = 0
        while j < b:
            Z[i][j] = policy(X[i][j], Y[i][j])
            j = j + 1
        i = i + 1

    img2 = plt.figure(fig_label)
    img2.clear()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    cmap = plt.get_cmap('RdBu')
    cs = plt.contourf(X, Y, Z, cmap=cmap, vmin=-4, vmax=4)
    cbar = plt.colorbar(ticks=[-4, 0, 4])
    cbar.set_label(z_label, rotation=270, labelpad=10)

    filename = (str(fig_label)).replace(" ", "")

    plt.savefig(filename, bbox_inches='tight', dpi=100)

    if display is True:
        img2.show()
        plt.draw()
    else:
        plt.close(img2)

    plt.pause(0.00001)


def show_policy(policy, fig_label, title, x_label, y_label, z_label):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-3, 3, 100)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    a = len(Z)
    b = len(Z[0])
    i = 0
    while i < a:
        j = 0
        while j < b:
            Z[i][j] = policy(X[i][j], Y[i][j])
            j = j + 1
        i = i + 1

    img2 = plt.figure(fig_label)
    img2.clear()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    cmap = plt.get_cmap('RdBu')
    cs = plt.contourf(X, Y, Z, cmap=cmap, vmin=-4, vmax=4)
    cbar = plt.colorbar(ticks=[-4, 0, 4])
    cbar.set_label(z_label, rotation=270, labelpad=10)

    img2.show()
    plt.draw()

    plt.pause(0.00001)


def store_fitted_q_model(model, action_space, fig_label, title, x_label, y_label, z_label, display=False):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-3, 3, 100)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    a = len(Z)
    b = len(Z[0])
    i = 0
    while i < a:
        j = 0
        while j < b:
            best_reward = -math.inf
            for move in action_space:
                tmp = np.array([X[i][j], Y[i][j], move]).reshape(1, -1)
                reward = model.predict(tmp)

                if reward > best_reward:
                    best_reward = reward

            Z[i][j] = best_reward
            j = j + 1
        i = i + 1

    store_graph(X, Y, Z, fig_label, title, x_label, y_label, z_label, display=display)


def show_fitted_q_model(model, action_space, fig_label, title, x_label, y_label, z_label):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-3, 3, 100)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    a = len(Z)
    b = len(Z[0])
    i = 0
    while i < a:
        j = 0
        while j < b:
            best_reward = - math.inf

            for move in action_space:
                tmp = np.array([X[i][j], Y[i][j], move]).reshape(1, -1)
                reward = model.predict(tmp)
                if reward > best_reward:
                    best_reward = reward

            Z[i][j] = best_reward
            j = j + 1
        i = i + 1

    show_graph(X, Y, Z, fig_label, title, x_label, y_label, z_label)


def show_graph(X, Y, Z, fig_label, title, x_label, y_label, z_label):
    img2 = plt.figure(fig_label)
    img2.clear()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    cmap = plt.get_cmap('RdBu')

    plt.contourf(X, Y, Z, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_label(z_label, rotation=270, labelpad=10)

    img2.tight_layout()
    img2.show()
    plt.draw()

    plt.pause(0.00001)


def store_graph(X, Y, Z, fig_label, title, x_label, y_label, z_label, display=False):
    img2 = plt.figure(fig_label)
    img2.clear()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    cmap = plt.get_cmap('RdBu')

    plt.contourf(X, Y, Z, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_label(z_label, rotation=270, labelpad=10)

    filename = (str(fig_label)).replace(" ", "")

    plt.savefig(filename, bbox_inches='tight', dpi=100)

    if display is True:
        img2.tight_layout()
        img2.show()
        plt.draw()

    else:
        plt.close(img2)

    plt.pause(0.00001)