import torch
import random
import numpy as np
from game import FlappyBirdAI
from collections import deque
from model import Linear_QNet,QTrainer
from helper import plot
import keyboard

MAX_MEMORY = 100_000
BATCH_SIZE = 500
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.6
        self.memory = deque(maxlen=MAX_MEMORY) # popleft
        self.model = Linear_QNet(9, 16 ,2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game):
        # [dist. from pole, height, y level of player] min, game.obstacle[indx].height,game.obstacle[(indx+1)%3].x-50,game.obstacle[(indx+1)%3].height
        state = [game.obstacle[game.point%3].x, game.obstacle[game.point%3].height, game.obstacle[game.point%3].height - 150 ,game.obstacle[(game.point+1)%3].x,game.obstacle[(game.point+1)%3].height, game.obstacle[(game.point+1)%3].height-150, game.y, game.acc, game.gravity]
        return np.array(state , dtype=float)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        for states, actions, rewards, next_states, dones in mini_sample:
            self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = 100 - self.n_games
        final_move = [0,0]
        
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state , dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        # [straight , jump]
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = FlappyBirdAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        # make prediction
        final_move = agent.get_action(state_old)

        reward,done,score = game.play(final_move)
        state_new = agent.get_state(game)
        
        # train short term
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            if agent.n_games % 75 == 0:
                agent.model.save_big(file_name='test7')

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            

if __name__ == '__main__':
    train()