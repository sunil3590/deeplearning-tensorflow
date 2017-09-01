# neural-nuts
Implement deep learning networks
<hr>

### 1_linear_regression
`python 1_linear_regression.py`

#### Network
The network is just a single neuron which learns a function of the form y=mx+c<br>
Neuron uses a Gradient Descent Optimizer, Squared Difference as cost function and linear activation

#### Data
Data is generated as (x, y) pairs according to an equation, these data points are used to train the network
<hr>

### 2_word_autoencoder
`python 2_word_autoencoder.py`

#### Network
The network is a simple multi layer perceptron with 1 hidden layer.<br>
It takes a 1 hot vector for a word and encodes it to a lower dimension space (input dimension / 4)<br>
The decoder then tries to reconstruct the embedding to the original word

#### Data
Text data is a sample paragraph from the internet
<hr>

### 3_rl_gamer
`python 3_rl_gamer.py`

[Reference](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)

This network solves the CartPole-v1 environment - [evaluation](https://gym.openai.com/evaluations/eval_36Cos6YGTbefVLyPH5VlMA)<br>
A reinforcement learning agent is implemented using PyTorch

#### Network
The network is a simple multi layer perceptron with 1 hidden layer.<br>
It takes the state of the game as provided by OpenAI gym<br>
<hr>
