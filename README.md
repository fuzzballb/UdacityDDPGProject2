
//////////////////////////////////////////
/// Work in progress this is a Readme from an other project i am usign as a template
/////////////////////////////////////////


# Udacity Reinforcement Learning Project2



## Introduction 

This is the second project of the Udacity Deep Reinforcement Learning course. In this project Udacity provides a Unity3D application that is used as a training environment with DDPG (Deep Deterministic Policy Gradient). The goal of this environment is for the robot arm to follow the green ball. The environment provides position and angles of the joints. The resulting vector is passed to the Jupyter Notebook as a state.

```python
Number of agents: 1
Number of actions: 4
States look like: [  0.00000000e+00  -4.00000000e+00   0.00000000e+00   1.00000000e+00
  -0.00000000e+00  -0.00000000e+00  -4.37113883e-08   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00  -1.00000000e+01   0.00000000e+00
   1.00000000e+00  -0.00000000e+00  -0.00000000e+00  -4.37113883e-08
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   5.75471878e+00  -1.00000000e+00
   5.55726671e+00   0.00000000e+00   1.00000000e+00   0.00000000e+00
  -1.68164849e-01]
States have length: 33
```

The agent tries to find the action with the most future cumulative reward, and thus trains the deep Neural network to predict the best action, given a random state.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/c-2tIOe-1K4/0.jpg)](https://www.youtube.com/watch?v=c-2tIOe-1K4). 

*Training in progress*

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/sxyRiDX6dzE/0.jpg)](https://www.youtube.com/watch?v=sxyRiDX6dzE)

*Trained robot arm*


## Setup the environment in Windows 10

As with most machine learning projects, its best to start with setting up a virtual environment. This way the packages that need to be imported, don't conflict with Python packages that are already installed. For this project i used the Anaconda environment based on Python 3.6. 

While the example project provides a requirements.txt, i ren into this error while adding the required packages to your project

```python
!pip -q install ./
ERROR: Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0) (from versions: 0.1.2, 0.1.2.post1, 0.1.2.post2)
ERROR: No matching distribution found for torch==0.4.0 (from unityagents==0.4.0)
```

The solution, is to install a downloaded wheel file form the PyTourch website yourself. I downloaded "torch-0.4.1-cp36-cp36m-win_amd64.whl" from the PyTourch site https://pytorch.org/get-started/previous-versions/

```
(UdacityRLProject1) C:\Clients\Udacity\deep-reinforcement-learning\[your project folder]>pip install torch-0.4.1-cp36-cp36m-win_amd64.whl
Processing c:\clients\udacity\deep-reinforcement-learning\[your project folder]\torch-0.4.1-cp36-cp36m-win_amd64.whl
Installing collected packages: torch
Successfully installed torch-0.4.1
```

After resolving the dependencies, i still had a code issue, because the action returned a numpy.int64 instead of an in32.

```
packages\unityagents\environment.py", line 322, in step
for brain_name in list(vector_action.keys()) + list(memory.keys()) + list(text_action.keys()):
AttributeError: 'numpy.int64' object has no attribute 'keys'
```

When all dependencies and issues are resolved, the training can begin.

## Training the agent with code provided by the course

To start, and make sure the environment works, I have used the DDPG example that was referred to by the training video. My first training result was just using the defaults from the example, and didn't perform at all. I rewread the paper but had to see what other students where encountered, before getting better results. 

![alt text](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/FailedToLearn.PNG "Training with default epsilon decay")


## Solutions for getting a better score

I found a few possible issues with using the default solution and tried them one by one.

1. reduce noise

The noise that is added to the training was to much so i reduced the sigma from 0.2 to 0.1. This alone did not do a lot for the training results

![alt text](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Result1.PNG "Changed sigma")


2. increase episode length

Next i found that the agent probely needed more time to get to it's goal then de maximum episode length that i specified. Thus the agent rarely got to it's goal. and if it dit, it was because it was aleady close. This was the first time the score excided 1.0. Unfortunately it got stuk at around 2.x. not nearly the 30 i needed

![alt text](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Result2.PNG "Changed episode length")


3. Normalize 

By defining batch normalisation and adding it to the forward pass

still around the 2.x once finished check if removing this matters

![alt text](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Result3.PNG "Normalisation")

4. Increase replay buffer size

[GPU] helped get above 3.x


5. resetting the agent after every

[LOCAL example] agent.reset() helped get above 3.x in the first 100 episodes without the previous increased buffer size step
 
6. learinig for >500 episodes

When learning for more then 500 episodes connection with the Udacity environment gets lost.  

Reloading saved wights to continue learning didn't work, probebly because i didn't save and reload the *target* networks of the actor and critic.
 
7. increasing learning rate 

if the steps in learning are to small, it can take a long time before the optimal value is found, make it to big, and you will overshoot your optimal value 

![alt text](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Result6.PNG "Learning rate")


## Learning Algorithm

1. First we initialize the agent in the Navigation notebook

   *Navigation.ipynb*

```Python
# initialise an agent
agent = Agent(state_size=33, action_size=4, random_seed=2)
```

2. This sets the state and action size for the agent and creates an Actor, and a Critic neural net, with corresponding target network. 

   *ddpg_agent.ipynb*

```Python
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
```

The critic network takes actions and states and produces a Q value. This is compaired to the actual value in the environment, the difference between expected Q and actual reward from the environment is used to calculate a loss, which it tries to minimize. When the critic starts giving estimates about the Q value given states and actions, the actor network can use these trained values, to train the best action for a given state. 

The target networks are there to train more gradually, because every step only a small potion of the differnace of the weights is copied from the source to the target network.

for a more detailed explination see the video below 

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/_pbd6TCjmaw/0.jpg)](https://www.youtube.com/watch?v=_pbd6TCjmaw&t=454). 


3. Adding noise to increase exploration instead of only exploiting the paths that have lead to success

   *ddpg_agent.ipynb*

```Python
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
```

Here we define noise that will be added to the action that is obtained from from the actor network.


4. Replay buffer, to store past experences

   *ddpg_agent.ipynb*

```Python
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
```

Becasue DDPG is a off policy algorithm (see bottom of the page), just like Q learning. It can learn from past experiences that are stored in the replay buffer. 









## on policy vs off policy

The on-policy aggression is like SARSA, they assume that their experience comes from the agents himself, and they try to improve agents policy right down this online stop. So, the agent plays and then it improves and plays again. And what you want to do is you want to get optimal Strategies as quickly as possible. 

Off-policy algorithms like Q-learning, as an example, they have slightly relax situation. They don't assume that the sessions you're obtained, you're training on, are the ones that you're going to use when the agent is going to finally get kind of unchained and applied to the actual problem. Because of this, it can use data from an experiance replay buffer 



## Model based vs model free 

Model-based reinforcement learning has an agent try to understand the world and create a model to represent it. Here the model is trying to capture 2 functions, the transition function from states $T$ and the reward function $R$. From this model, the agent has a reference and can plan accordingly.

However, it is not necessary to learn a model, and the agent can instead learn a policy directly using algorithms like Q-learning or policy gradient.

A simple check to see if an RL algorithm is model-based or model-free is:

If, after learning, the agent can make predictions about what the next state and reward will be before it takes each action, it's a model-based RL algorithm.
