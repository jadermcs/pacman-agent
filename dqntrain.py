import layout
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from pacman import ClassicGameRules

BATCH = 512
LR = 0.001
EPOCH = 10000
memory = []
MAX_MEM = 10000
index2action = {x:y for (x,y) in\
                  enumerate(['North', 'South', 'East', 'West', 'Stop'])}
action2index = {y:x for (x,y) in index2action.items()}

class CNNModel(nn.Module):
    def __init__(self, width, height):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.flatten = nn.Flatten()
        size = (width-2*3)*(height-2*3)*32
        self.linear1 = nn.Linear(size, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


def color(char):
    if char == '%':# wall
        return [0, 0, 255]
    elif char == 'o':# capsule
        return [255, 255, 255]
    elif char == '.':# food
        return [155, 155, 155]
    elif char == 'G':# food
        return [255, 0, 0]
    elif char == '<' or char == '>' or char == 'v' or char == '^':# pacman
        return [255, 255, 0]
    else:# nothing
        return [0, 0, 0]

def convert_to_image(state):
    text = str(state.data).split('\n')[:-2]
    width, height = len(text[0]), len(text)
    image = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(width):
        for j in range(height):
            image[j,i] = color(text[j][i])
    return image

def train(data, model, loss_fn, optimizer, device):
    x, y = data
    tensor = torch.tensor(np.moveaxis(x, -1, 1)/255.0,
                          dtype=torch.float).to(device)
    pred = model(tensor)
    y = torch.tensor(y, dtype=torch.float).to(device)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main(layout, display):
    import pacmanAgents, ghostAgents, textDisplay
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    laystr = str(layout).split('\n')
    model = CNNModel(len(laystr[0]), len(laystr)).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    win_cnt = 0
    EPSILON = 0.5
    for e in range(EPOCH):
        GAMMA = 0.7
        rules = ClassicGameRules()
        agents = [pacmanAgents.DQNAgent()] +\
        [ghostAgents.RandomGhost(i+1) for i in range(layout.getNumGhosts())]
        if display:
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False
        game = rules.newGame(layout, agents[0], agents[1:], gameDisplay)
        state = game.state
        done = False
        loss = 0.0
        count = 0
        while not done:
            image = convert_to_image(state)
            # fig = plt.figure()
            # ax = fig.add_subplot(1, 1, 1)
            # ax.imshow(image.astype(int))
            # plt.show()
            EPSILON *= .99
            EPSILON = max(0.1, EPSILON)
            GAMMA *= .7
            GAMMA = max(0.01, GAMMA)
            if np.random.uniform() <= EPSILON:
                action = np.random.choice(state.getLegalPacmanActions())
            else:
                tensor = torch.tensor(np.moveaxis(image, -1, 0)/255.0,
                                      dtype=torch.float).unsqueeze(0).to(device)
                q = model(tensor)
                action = np.argsort(q[0].cpu().detach().numpy())
                action = [index2action[x] for x in action if index2action[x] in\
                          state.getLegalPacmanActions()][0]

            last_state = convert_to_image(state)
            reward = state.data.score
            for i, a in enumerate(agents):
                if i == 0: state = state.generateSuccessor(i, action)
                else: state = state.generateSuccessor(i, a.getAction(state))
                done = state.isWin() or state.isLose()
                if done: break
            rules.process(state, game)
            reward = state.data.score - reward
            if state.isWin():
                win_cnt += 1
            memory.append([[last_state, action2index[action], reward,
                            convert_to_image(state)], done])
            lenmem = len(memory)
            if lenmem > MAX_MEM:
                del memory[0]
            inputs = np.zeros((min(lenmem, BATCH), image.shape[0], image.shape[1],
                               3), dtype=float)
            targets = np.zeros((inputs.shape[0], len(action2index)),
                               dtype=float)
            for i, idx in enumerate(np.random.randint(0, max(lenmem-1, 1),
                                                      size=inputs.shape[0])):
                (obs0, action0, reward0, obs1), game_over = memory[idx]
                inputs[i:i+1] = obs0
                tensor = torch.tensor(np.moveaxis(obs0, -1, 0)/255.0,
                                      dtype=torch.float).unsqueeze(0).to(device)
                targets[i] = model(tensor)[0].cpu().detach().numpy()
                tensor = torch.tensor(np.moveaxis(obs1, -1, 0)/255.0,
                                      dtype=torch.float).unsqueeze(0).to(device)
                q_sa = np.max(model(tensor)[0].cpu().detach().numpy())
                if game_over:
                    targets[i, action0] = reward0
                else:
                    targets[i, action0] = reward0 + GAMMA * q_sa
            loss += train((inputs, targets), model, loss_fn, optimizer, device)
            count += 1
        print(f"epoch: {e:04d} lr:{LR:.4f} win_count: {win_cnt:03d} episodes: {count:03d} "
              f"loss: {loss/count:>7f} ")

if __name__ == "__main__":
    from optparse import OptionParser
    usageStr = """
    USAGE:      python dqntrain.py <options>
    EXAMPLES:   (1) python dqntrain.py
                    - starts an interactive game
                (2) python pacman.py --layout smallClassic --zoom 2
                OR  python pacman.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = OptionParser(usageStr)
    parser.add_option('--layout', dest='layout',
                      help='the LAYOUT_FILE',
                      metavar='LAYOUT_FILE', default='mediumClassic')
    parser.add_option('--display', action='store_true', dest='display',
                      help='display output', default=False)

    options, args = parser.parse_args()
    main(layout.getLayout(options.layout), options.display)
