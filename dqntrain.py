import layout
import numpy as np
import torch
from torch import nn
# import matplotlib.pyplot as plt
from pacman import ClassicGameRules

BATCH = 64
LR = 0.0001
EPOCH = 10000
EPSILON = 0.5
memory = list()

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 8, 2, stride=2)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(224, 32)
        self.linear2 = nn.Linear(32, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
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
    return np.moveaxis(image, -1, 0)/255.0

def train(layout, display):
    import pacmanAgents, ghostAgents, textDisplay
    rules = ClassicGameRules()
    if display:
        gameDisplay = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        gameDisplay = display
        rules.quiet = False
    agents = [pacmanAgents.DQNAgent()] +\
        [ghostAgents.RandomGhost(i+1) for i in range(layout.getNumGhosts())]
    game = rules.newGame(layout, agents[0], agents[1:], gameDisplay)
    state = game.state
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNModel().to(device)
    done = False
    while not done:
        if np.random.uniform() <= EPSILON:
            continue
            action = np.random.choice()
        else:
            image = convert_to_image(state)
            image = torch.tensor(image).unsqueeze(0).to(device)
            q = model(image)
            action = np.argmax(q[0].cpu().numpy())
            done = True
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(image)
    # plt.show()

if __name__ == "__main__":
    from optparse import OptionParser
    usageStr = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python pacman.py
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
    train(layout.getLayout(options.layout), options.display)
