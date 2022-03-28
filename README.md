# battleship-ai
Playground playing battleship with deep learning and other fun algorithms

As of now, We didnt get it working for successfully playing more than 1 board with RL.

I did do quite a bit of research and just incase I come back to this in the future to try again, I wanted to add a couple of notes to make it more doable. 

I did get a couple different implementations of Deep Q Networks working for OpenAI gym LunarLander-v2 working. And a visual solution of the cartpole possibly working with a Convolutional DQN maybe working.  That code is all in the rl_practice directory.

In the main directory, The battleship.py file is the implementation of the battleship game that can be plugged into a network. 

Then I have 4 different networks that I tried to solve with. the ones that are just called dqn_battleship and wrapper and main are the fully linear 1d attempt. So the state and action space are both 1d with 100 options. That is also the final version that I came back to at the end to try some of my final ideas like having the reward function incentivize firing near previous hits (I think that is a good idea). I also have some notes for next steps with that one. 

Then the files that have "conv" in the title was an attempt to do a 2d state space (10x10) with a convolutional network and a 2d action space (2x10) for selecting rows and columns. I think that simplifying the action space to 2x10 caused more issues than it solved. Because it tended to pick everything from a row before moving on. 

So after that, I did the the ones with "3" in them. This is also using a 2d state space (10x10) with a convolutional network. But I left the action space as a 1x100 output. I think this one probably has the most promise of working. I think doing a convolution on the state space will help with spacial awareness. But it will need some of the later features added to the first file sets to work. Plus maybe more

The last set of files (with a "4") were another attempt but with a different implementation of the the Q network. from another source. This runs much slower than the other the original and "3" setups. And it still doesnt solve it anyways. But maybe worth having the code just in case. 

Here are the package installs that I used to get the code working. 
```
pip3 install torch gym pyopengl glu pyglet Box2D pandas torchvision
apt install python-opengl swig python-pygame
```

Here are some links to some of the other battle ship and relevant attempts that I found online

This one claims to have solved it, but does not provide any source code. But they do have a couple links to things that they said they used for the network. 
https://www.ga-ccri.com/deep-reinforcement-learning-win-battleship

Thess are some of the sources from the above article
https://github.com/spro/practical-pytorch/blob/master/reinforce-gridworld/reinforce-gridworld.ipynb
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


This one looks like it solved a simplified version of battleship, but it doesnt look like they solved the whole game so they may be on to something
https://towardsdatascience.com/an-artificial-intelligence-learns-to-play-battleship-ebd2cf9adb01

This is the place where I found the implementation of the lunar lander that worked the best. This is what most of my implementations of the battleship solution are built around
https://www.katnoria.com/nb_dqn_lunar/

Reddit thread that seemed to come to a similar solution as me. That no one has solved it completely and provided source code. 
https://www.reddit.com/r/learnmachinelearning/comments/ab2jdh/battleships_reinforcement_learning/

The last idea I had that I have not implemented yet is to do a separate learn function that is similar to how it runs right now, but rather than just using random samples from history to try to train the network, use a single state and analyze the reward return from taking every action from that state, and see if that produces any benefit. 

I think that likely what this will take is 2 seperate networks. One trained to find ships when there are no active unsunk ships. And then one that uses a much smaller local state around a hit to sink the rest of the boat after a hit is found. 

It seemed like the state and action space was just too large for it to solve and start the learing with the current implementation

