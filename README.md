
//////////////////////////////////////////
/// Work in progress this is a Readme from an other project i am usign as a template
/////////////////////////////////////////


# Udacity Reinforcement Learning Project2



## Introduction 

This is the second project of the Udacity Deep Reinforcement Learning course. In this project Udacity provides a Unity3D application that is used as a training environment with DDPG. The goal of this environment is for the robot arm to follow the green ball. The environment provides position and angles of the joints. The resulting vector is passed to the Jupyter Notebook as a state.

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


{Update image with initial training results}
![alt text](https://github.com/fuzzballb/UdacityRFlearningProject1/blob/master/images/Eps_decay_0_995.PNG "Training with default epsilon decay")


## Solutions for getting a better score

I found a few possible issues with using the default solution and tried them one by one.

1. reduce noise

The noise that is added to the training was to much so i reduced the sigma from 0.2 to 0.1. This alone did not do a lot for the training results

2. increase episode length

Next i found that the agent probely needed more time to get to it's goal then de maximum episode length that i specified. Thus the agent rarely got to it's goal. and if it dit, it was because it was aleady close. This was the first time the score excided 1.0. Unfortunately it got stuk at around 2.x. not nearly the 30 i needed

3. Normalize 

By defining batch normalisation and adding it to the forward pass

still around the 2.x once finished check if removing this matters

4. Increase replay buffer size

[GPU] helped get above 3.x

5. resetting the agent after every

[LOCAL example] agent.reset() helped get above 3.x in the first 100 episodes without the 4. increased buffer size
 
6. 

issues with learning more then 500 episodes connection gets lost

Reloading saved wights didn't work, probebly because i didn't save and reload the target networks of the actor and critic

7. increasing learning rate 

if the steps in learning are to small, it can take a long time before the optimal value is found

