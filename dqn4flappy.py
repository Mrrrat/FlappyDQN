import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append("game/")
import random
import numpy as np
import flappy_bird_gym
from collections import deque
from keras.layers import Input, Dense
from keras.models import load_model, Sequential
from keras.optimizers import RMSprop


def NeuralNet(input_shape, output_shape):
    model = Sequential()
    model.add(Input(input_shape))
    model.add(Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform'))
    model.add(Dense(256, activation="relu", kernel_initializer='he_uniform'))
    model.add(Dense(64, activation="relu", kernel_initializer='he_uniform'))
    model.add(Dense(output_shape, activation="linear", kernel_initializer='he_uniform'))
    model.compile(loss="mse", optimizer=RMSprop(lr=0.0003, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model


class DQNAgent:
    def __init__(self):
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.episodes = 1000
        self.state_n = self.env.observation_space.shape[0]
        self.action_n = self.env.action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.min_eps = 0.001
        self.eps = 1.0
        self.eps_k = 0.9999
        self.batch_n = 64
        self.train_start = 1000
        self.jump_prob = 0.01
        self.model = NeuralNet(input_shape=(self.state_n,), output_shape=self.action_n)

    def act(self, state):
        if np.random.random() > self.eps:
            return np.argmax(self.model.predict(state))
        return 1 if np.random.random() < self.jump_prob else 0

    def learn(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_n))
        state = np.zeros((self.batch_n, self.state_n))
        next_state = np.zeros((self.batch_n, self.state_n))
        action, reward, done = [], [], []
        for i in range(self.batch_n):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        for i in range(self.batch_n):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
        self.model.fit(state, target, batch_size=self.batch_n, verbose=0)

    def train(self):
        for i in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_n])
            done = False
            score = 0
            self.eps = self.eps * self.eps_k if self.eps * self.eps_k > self.min_eps else self.min_eps
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_n])
                score += 1
                if done:
                    reward -= 100
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                if done:
                    print("Episode: {}, Score: {}, e: {:.2}".format(i, score, self.eps))
                    if score >= 1000:
                        self.model.save("flappy-bird.h5")
                        return
                self.learn()

    def perform(self):
        self.model = load_model("flappy-bird.h5")
        while 1:
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_n])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_n])
                score += 1
                if done:
                    print("Score: {}".format(score))
                    break


if __name__ == "__main__":
    agent = DQNAgent()
    #agent.train()
    #agent.perform()
