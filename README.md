# INFO8003-1 - Optimal decision making for complex problems
## Final project
### Running the code
This file comments on how to use the code done in the first assignment. In order to do so, one should run the following :
```sh
$ python3 run.py <options>
```
When called without option, the code runs without training, for a single episode and with a epsilon of 0. In order to change that, one can use the following parameters:
* `--epsilon` **float**: Initial value of the Epsilon parameter for the decaying e-greedy policy and the OU process. Should be between 0 and 1.
* `--nb_episodes` **int** : Number of episodes (>0).
* `--train` **{0, 1}** : 1 if the user wants to retrain the network. 0 otherwise
* `--video` **{0, 1}** : 1 if the user wants to create a video of the simulation. 0 otherwise
* `--plot` **{0, 1}** : 1 if the user wants to make plot of the simulation. 0 otherwise

### Training the agent
In order to train the agent, one should put the `train` argument to 1. 
If the user wants to train from scratch, he should delete the pre-existing weights of the models : *actor_w.h5* and *critic_W.h5*.
