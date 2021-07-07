import torch
import argparse
import layout
import numpy as np
# import matplotlib.pyplot as plt
from pacman import ClassicGameRules

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
    image = np.zeros((width, height, 3), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            image[i,j] = color(text[j][i])
    return image


def train(layout, display):
    import pacmanAgents, ghostAgents, textDisplay
    rules = ClassicGameRules()
    if display:
        gameDisplay = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        gameDisplay = display
        rules.quiet = False
    agents = [pacmanAgents.DQNAgent()] + [ghostAgents.RandomGhost(i+1) for i in range(layout.getNumGhosts())]
    game = rules.newGame(layout, agents[0], agents[1:], gameDisplay)
    state = game.state
    image = convert_to_image(state)
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
