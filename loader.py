import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import plot
import numpy as np
from game import FlappyBirdAI

# Define the model architecture
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.itr = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear3(x))
        x = self.linear2(x)
        return x

# Create an instance of the model
model = Linear_QNet(input_size=9, hidden_size=16, output_size=2)

# Load the model's state_dict from a saved file
model.load_state_dict(torch.load('./model/max_49.pth'))

def get_state(game):
    # [dist. from pole, height, y level of player] min, game.obstacle[indx].height,game.obstacle[(indx+1)%3].x-50,game.obstacle[(indx+1)%3].height
    state = [game.obstacle[game.point%3].x, game.obstacle[game.point%3].height, game.obstacle[game.point%3].height - 150 ,game.obstacle[(game.point+1)%3].x,game.obstacle[(game.point+1)%3].height, game.obstacle[(game.point+1)%3].height-150, game.y, game.acc, game.gravity]
 
    # state = [game.obstacle[game.point%3].x, game.obstacle[game.point%3].height -5, game.obstacle[game.point%3].height - 155 ,game.obstacle[(game.point+1)%3].x,game.obstacle[(game.point+1)%3].height, game.obstacle[(game.point+1)%3].height-150, game.y]
    return np.array(state , dtype=float)

def get_action(state):
        final_move = [0,0]
        
        state0 = torch.tensor(state , dtype=torch.float)
        prediction = model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        # [straight , jump]
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    n_games = 0
    game = FlappyBirdAI()
    while True:
        # get old state
        state_old = get_state(game)
        # make prediction
        final_move = get_action(state_old)

        reward,done,score = game.play(final_move)

        if done:
            game.reset()
            n_games += 1

            if score > record:
                record = score
            

            print( 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            

if __name__ == '__main__':
    train()

