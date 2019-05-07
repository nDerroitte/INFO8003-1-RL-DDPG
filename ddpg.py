from replay import *
import numpy as np
from OU import *

# DDPG CST
BUFFER_SIZE = 100000    # Max size of the history
BATCH_SIZE = 32         # Size of the mini-batch
GAMMA = 0.99            # Discount factor
TAU = 0.01              # Target Network Learning rate
LRA = 0.001             # Learning rate for actor
LRC = 0.001             # Lerning rate for critic
action_dim = 1          # Move
state_dim = 4           # Bar : x,vx Fruit : x,y
OU = OU()               # Ornstein-Uhlenbeck Process

def load_weights(actor, critic):
    """
    Load the weights of the networks if they exist

    Parameters:
    ----------
    actor : Actor network (actor object)
    critic: Critic network (critic object)
    """
    try:
        actor.model.load_weights('actor_w.h5')
    except:
        print("No 'actor_w.h5' file found, using new weights.")
    try:
        critic.model.load_weights('critic_w.h5')
    except:
        print("No 'critic_w.h5' file found, using new weights.")

def save_weights(actor, critic, safe=False):
    """
    Safe the weights of the networks

    Parameters:
    ----------
    actor : Actor network (actor object)
    critic: Critic network (critic object)
    safe : boolean, optional
        If true, safe the network under an other name/
    """
    if safe:
        actor.model.save_weights('actor_wS.h5')
        critic.model.save_weights('critic_wS.h5')
    else:
        actor.model.save_weights('actor_w.h5')
        critic.model.save_weights('critic_w.h5')

def getAction(sess, states, actor, train_indicator, epsilon, action_dim, EPS):
    """
    Get the action from the Actor model.

    Parameters:
    -----------
    sess : A tensorflow session
    states : []
        List of the 4 elements composing the states
    actor : Actor network (actor object)
    train_indicator : boolean
        1 if train in enable, 0 otherwise
    epsilon : float
        Max value of epsilon
    action_dim : int
        Dimension of the action-space
    EPS: float
        current value of epsilon
    """
    # Variable init
    a_t = np.zeros([1,action_dim])
    noise_t = np.zeros([1,action_dim])
    # Action prediction
    a_t_original = actor.model.predict(states.reshape(1, states.shape[0]))

    if train_indicator and EPS > random.random():
        # Epsilon GREEDY
        a_t[0][0] = random.uniform(-5, 5)
        # OU PROCESS
        noise_t[0][0] = train_indicator * max(epsilon, 0) * (
                        OU.function(a_t_original[0][0],  0.0 , 0.75, 0.1))
        a_t[0][0] = a_t[0][0] + noise_t[0][0]
    else:
        a_t[0][0] = a_t_original[0]
    return a_t

def trainModels(buff, sess, st, r, st1, a, loss, actor, critic):
    """
    Train the critic model.

    Parameters:
    -----------
    buff : replay object
        History
    sess : A tensorflow session
    st : []
        List of the 4 elements composing the state
        current state
    r : double
        Value of the reward fct
    st1 : []
        List of the 4 elements composing the state
        Next state state
    a : double
        Value of the action
    loss: double
        Value of the loss function of the critic network
    actor : Actor network (actor object)
    critic : Critic network (critic object)
    """
    buff.add(st, a, r, st1)

    # Get experiences from the batch
    batch = buff.getBatch(BATCH_SIZE)
    states = np.asarray([e[0] for e in batch])
    actions = np.asarray([e[1] for e in batch])
    rewards = np.asarray([e[2] for e in batch])
    new_states = np.asarray([e[3] for e in batch])

    y_t = np.asarray([e[1] for e in batch])
    # Compute the value of the Q from the critic
    target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
    # Compute the true y
    for k in range(len(batch)):
        y_t[k] = rewards[k] + GAMMA*target_q_values[k]
    # Update the critic
    loss += critic.model.train_on_batch([states,actions], y_t)
    # Get the action
    a_for_grad = actor.model.predict(states)
    # Get the gradient for theses action
    grads = critic.gradients(states, a_for_grad)
    # Update the w of the actor
    actor.train(states, grads)
    # Update the w of the target networks
    actor.target_train()
    critic.target_train()
    return loss
