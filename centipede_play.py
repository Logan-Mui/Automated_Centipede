import argparse
import random
import sys
import pdb
import time

import gymnasium as gym
import numpy
from gymnasium import wrappers, logger
import numpy as np
from scipy import stats

# (y, x)

# starting pos of the elf
current_y = 160
current_x = 48
startBoard = {
    "BLACK": (0, 0, 0),
    "BAR": (110, 156, 66),
    "SCORE": (188, 144, 252),
    "BRICK": (181, 83, 40),
    "CENTIPEDE": (184, 70, 162),
    "SPIDER": (146, 70, 192)
}

anotherBoard = {
    "BLACK": (0, 0, 0),
    "BAR": (66, 114, 194),
    "SCORE": (188, 144, 252),
    "BRICK": (45, 50, 194),
    "CENTIPEDE": (184, 50, 50),
    "SPIDER": (110, 156, 66)
}

board3 = {
    "BLACK": (0, 0, 0),
    "BAR": (198, 108, 58),
    "SCORE": (188, 144, 252),
    "BRICK": (187, 187, 53),
    "CENTIPEDE": (146, 70, 192),
    "SPIDER": (84, 138, 210)
}

board4 = {
    "BLACK": (0, 0, 0),  # always the same
    "BAR": (66, 72, 200),
    "SCORE": (188, 144, 252),  # always the same
    "BRICK": (184, 70, 162),
    "CENTIPEDE": (110, 156, 66),
    "SPIDER": (181, 83, 40)
}

boards = [startBoard, anotherBoard, board3, board4]
boardIndex = 0
currBrick = boards[0].get("BRICK")

ELF_WIDTH = 30
ELF_HEIGHT = 20
ELF_START_TOP_LEFT = (160, 48)
ELF_START_BOTTOM_RIGHT = (180, 79)


# elf starts within (160, 48) - (180, 79)


def get_color_values():
    # should ideally do once each board and get correct board
    # it's slow to loop through each color to get color each time
    new_array = observation.reshape((observation.shape[0] * observation.shape[1], 3))
    colors = {tuple(x) for x in new_array}
    print(colors)


def find_instance(observation, target):
    result = zip(*np.where(observation == target)[:2])
    return [x for x in result]


def go_wild():
    return random.randint(10, 17)


class Agent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    # You should modify this function
    # You may define additional functions in this
    # file or others to support it.

    def act(self, observation, reward, done):
        """Compute and action based on the current state.
        
        Args:
            observation: a 3d array of the game screen pixels.
                Format: rows x columns x rgb.
            reward: the reward associated with the current state.
            done: whether or not it is a terminal state.

        Returns:
            A numerical code giving the action to take. See
            See the Actions table at:
            https://gymnasium.farama.org/environments/atari/centipede/
        """
        # observation_arr = np.array(observation)
        # observation_arr.nonzero()

        global current_y
        global current_x
        global boardIndex

        # the brick color changes every game - if brick color is not the same, switch boards

        if not find_instance(observation, boards[boardIndex].get("BRICK")):
            if boardIndex == 3:
                # usually dies board 4 but if it gets to a better level, turn into random agent
                # don't really wanna do more so just return random after level 4 completes lol
                return go_wild()
            else:
                boardIndex += 1
            print(boardIndex)
            get_color_values()

        centResult = find_instance(observation, boards[boardIndex].get("CENTIPEDE"))
        spiderResult = find_instance(observation, boards[boardIndex].get("SPIDER"))

        # print("curr positions of elf: (%d, %d)" % (current_y, current_x))

        # try to find where the centipede and spider are and shoot when y coor match stationary elf
        # need to make a function that fires when giving it the elf's y coor

        cXVals = []
        cYVals = []
        for c in centResult:
            cXVals.append(c[1])
            cYVals.append(c[0])

        sXVals = []
        sYVals = []
        for s in spiderResult:
            sXVals.append(s[1])
            sYVals.append(s[0])

        # pretty much always tracking the "most notable" value
        spidYCoor = stats.mode(sYVals)[0]
        spidXCoor = stats.mode(sXVals)[0]
        centYCoor = stats.mode(cYVals)[0]
        centXCoor = stats.mode(cXVals)[0]

        # right now the firing buffer is the width of the elf, new should be just around 5 or so px wide
        # also move buffer is 20px left or right of elf body
        if current_x < spidXCoor < current_x + 10:
            return 1
        elif current_x < centXCoor < current_x + 10:
            return 1
        # move toward centipede
        elif current_x - 5 > centXCoor:
            current_x -= 10
            return 12
        elif current_x + 25 < centXCoor:  # elf body from top left is 30px wide
            current_x += 10
            return 11
        # move away from spider
        elif current_x - 5 > spidXCoor:
            current_x += 10
            return 11
        elif current_x + 25 < spidXCoor:  # elf body from top left is 30px wide
            current_x -= 10
            return 12

        # not focusing on the x coordinates rn

        # elif current_y + 10 < spidYCoor:
        #     current_y += 10
        #     return 13
        # elif current_y + 40 > spidYCoor:
        #     current_y -= 10
        #     return 10
        # elif current_y + 10 < centYCoor:
        #     current_y += 10
        #     return 13
        # elif current_y + 40 > centYCoor:
        #     current_y -= 10
        #     return 10

        return 0


## YOU MAY NOT MODIFY ANYTHING BELOW THIS LINE OR USE
## ANOTHER MAIN PROGRAM
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', nargs='?', default='Centipede-v4', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id, render_mode="human")

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'random-agent-results'

    env.unwrapped.seed(0)
    agent = Agent(env.action_space)

    episode_count = 100
    reward = 0
    terminated = False
    score = 0
    special_data = {}
    special_data['ale.lives'] = 3
    observation = env.reset()[0]

    while not terminated:
        action = agent.act(observation, reward, terminated)
        time.sleep(.05)
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()

    # Close the env and write monitor result info to disk
    print("Your score: %d" % score)
    env.close()
