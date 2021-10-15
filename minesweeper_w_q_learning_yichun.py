#!/usr/bin/env python
# coding: utf-8

# In[31]:


# AUTHOR: Yichun Zhao

# !pip install mss
# !pip install tkinter
# !pip install IPython
# !python -m pip uninstall rl --yes
#!pip install pygame

#!pip install keras-rl2
#!pip install pyautogui

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

import pygame
import pygame.locals
import abc
import time
import numpy as np


class Visualizer(abc.ABC):

    @abc.abstractmethod
    def start(self, width, height):
        pass

class MinesweeeperVisualizer(Visualizer):
    TILE_SIZE = 16
    COLOUR_GREY = (189, 189, 189)
    #TILES_FILENAME = os.path.join(os.path.dirname(__file__), 'tiles.png')
    TILES_FILENAME = './tiles.png'
    TILE_HIDDEN = 9
    TILE_EXPLODED = 10
    TILE_BOMB = 11
    TILE_FLAG = 12
    WINDOW_NAME = 'Minesweeper'

    def __init__(self):
        self.game_width = 0
        self.game_height = 0
        self.num_mines = 0
        self.screen = None
        self.tiles = None

    def start(self, width, height, num_mines):
        self.game_width = width
        self.game_height = height
        self.num_mines = num_mines
        pygame.init()
        pygame.mixer.quit()
        pygame.display.set_caption(self.WINDOW_NAME)
        screen_width = self.TILE_SIZE * self.game_width
        screen_height = self.TILE_SIZE * self.game_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.screen.fill(self.COLOUR_GREY)
        self.tiles = self._load_tiles()

    def wait(self):
        while 1:
            event = pygame.event.wait()
            if event.type == pygame.locals.KEYDOWN:
                break
            elif event.type == pygame.locals.QUIT:
                pygame.quit()
                break

    def close(self, pause):
        if pause:
            self.wait()
        pygame.quit()

    def _load_tiles(self):
        image = pygame.image.load(self.TILES_FILENAME).convert()
        image_width, image_height = image.get_size()
        tiles = []
        for tile_x in range(0, image_width // self.TILE_SIZE):
            rect = (tile_x * self.TILE_SIZE, 0, self.TILE_SIZE, self.TILE_SIZE)
            tiles.append(image.subsurface(rect))
        return tiles

    def _draw(self, observation):
        openable = self.game_width * self.game_height - self.num_mines
        unique, counts = np.unique(observation, return_counts=True)
        unopened = dict(zip(unique, counts))[-1]
        all_opened = unopened == self.num_mines

        for x in range(self.game_width):
            for y in range(self.game_height):
                if observation[x, y] == -1:
                    if all_opened:
                        tile = self.tiles[self.TILE_BOMB]
                    else:
                        tile = self.tiles[self.TILE_HIDDEN]
                elif observation[x, y] == -2:
                    tile = self.tiles[self.TILE_EXPLODED]
                else:
                    tile = self.tiles[int(observation[x, y])]
                self.screen.blit(tile, (16 * x, 16 * y))
        pygame.display.flip()


# In[3]:


import sys
from six import StringIO
import random
from random import randint

import numpy as np
import gym
from gym import spaces

# default : easy board
BOARD_SIZE = 4
NUM_MINES = 4

# cell values, non-negatives indicate number of neighboring mines
MINE = -1
CLOSED = -2


def board2str(board, end='\n'):
    """
    Format a board as a string
    Parameters
    ----
    board : np.array
    end : str
    Returns
    ----
    s : str
    """
    s = ''
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            s += str(board[x][y]) + '\t'
        s += end
    return s[:-len(end)]


def is_new_move(my_board, x, y):
    """ return true if this is not an already clicked place"""
    return my_board[x, y] == CLOSED


def is_valid(x, y):
    """ returns if the coordinate is valid"""
    return (x >= 0) & (x < BOARD_SIZE) & (y >= 0) & (y < BOARD_SIZE)


def is_win(my_board):
    """ return if the game is won """
    return np.count_nonzero(my_board == CLOSED) == NUM_MINES


def is_mine(board, x, y):
    """return if the coordinate has a mine or not"""
    return board[x, y] == MINE


def place_mines(board_size, num_mines):
    """generate a board, place mines randomly"""
    mines_placed = 0
    board = np.zeros((board_size, board_size), dtype=int)
    while mines_placed < num_mines:
        rnd = randint(0, board_size * board_size)
        x = int(rnd / board_size)
        y = int(rnd % board_size)
        if is_valid(x, y):
            if not is_mine(board, x, y):
                board[x, y] = MINE
                mines_placed += 1
    return board

def to_s(row, col):
    return row*ncol + col


class MinesweeperDiscreetEnv(gym.Env):
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, board_size=BOARD_SIZE, num_mines=NUM_MINES):
        """
        Create a minesweeper game.
        Parameters
        ----
        board_size: int     shape of the board
            - int: the same as (int, int)
        num_mines: int   num mines on board
        """

        self.board_size = board_size
        self.num_mines = num_mines
        self.board = place_mines(board_size, num_mines)
        self.my_board = np.ones((board_size, board_size), dtype=int) * CLOSED
        self.num_actions = 0

        self.observation_space = spaces.Box(low=-2, high=9,
                                            shape=(self.board_size, self.board_size), dtype=np.int)
        self.action_space = spaces.Discrete(self.board_size*self.board_size)
        self.valid_actions = np.ones((self.board_size * self.board_size), dtype=np.bool)

    def count_neighbour_mines(self, x, y):
        """return number of mines in neighbour cells given an x-y coordinate
            Cell -->Current Cell(row, col)
            N -->  North(row - 1, col)
            S -->  South(row + 1, col)
            E -->  East(row, col + 1)
            W -->  West(row, col - 1)
            N.E --> North - East(row - 1, col + 1)
            N.W --> North - West(row - 1, col - 1)
            S.E --> South - East(row + 1, col + 1)
            S.W --> South - West(row + 1, col - 1)
        """
        neighbour_mines = 0
        for _x in range(x - 1, x + 2):
            for _y in range(y - 1, y + 2):
                if is_valid(_x, _y):
                    if is_mine(self.board, _x, _y):
                        neighbour_mines += 1
        return neighbour_mines

    def open_neighbour_cells(self, my_board, x, y):
        """return number of mines in neighbour cells given an x-y coordinate
            Cell -->Current Cell(row, col)
            N -->  North(row - 1, col)
            S -->  South(row + 1, col)
            E -->  East(row, col + 1)
            W -->  West(row, col - 1)
            N.E --> North - East(row - 1, col + 1)
            N.W --> North - West(row - 1, col - 1)
            S.E --> South - East(row + 1, col + 1)
            S.W --> South - West(row + 1, col - 1)
        """
        for _x in range(x-1, x+2):
            for _y in range(y-1, y+2):
                if is_valid(_x, _y):
                    if is_new_move(my_board, _x, _y):
                        my_board[_x, _y] = self.count_neighbour_mines(_x, _y)
                        if my_board[_x, _y] == 0:
                            my_board = self.open_neighbour_cells(my_board, _x, _y)
        return my_board

    def get_next_state(self, state, x, y):
        """
        Get the next state.
        Parameters
        ----
        state : (np.array)   visible board
        x : int    location
        y : int    location
        Returns
        ----
        next_state : (np.array)    next visible board
        game_over : (bool) true if game over
        """
        my_board = state
        game_over = False
        if is_mine(self.board, x, y):
            my_board[x, y] = MINE
            game_over = True
        else:
            my_board[x, y] = self.count_neighbour_mines(x, y)
            if my_board[x, y] == 0:
                my_board = self.open_neighbour_cells(my_board, x, y)
        self.my_board = my_board
        return my_board, game_over

    def reset(self):
        """
        Reset a new game episode. See gym.Env.reset()
        Returns
        ----
        next_state : (np.array, int)    next board
        """
        self.my_board = np.ones((self.board_size, self.board_size), dtype=int) * CLOSED
        self.board = place_mines(self.board_size, self.num_mines)
        self.num_actions = 0
        self.valid_actions = np.ones((self.board_size * self.board_size), dtype=bool)

        return self.my_board

    def step(self, action):
        """
        See gym.Env.step().
        Parameters
        ----
        action : np.array    location
        Returns
        ----
        next_state : (np.array)    next board
        reward : float        the reward for action
        done : bool           whether the game end or not
        info : {}             {'valid_actions': valid_actions} - a binary vector,
                                where false cells' values are already known to observer
        """
        state = self.my_board
        x = int(action / self.board_size)
        y = int(action % self.board_size)
        # print("-----")
        # print(action)
        # print(x)
        # print(y)

        # test valid action - uncomment this part to test your action filter if needed
        # if bool(self.valid_actions[action]) is False:
        #    raise Exception("Invalid action was selected! Action Filter: {}, "
        #                    "action taken: {}".format(self.valid_actions, action))

        next_state, reward, done, info = self.next_step(state, x, y)
        self.my_board = next_state
        self.num_actions += 1
        self.valid_actions = (next_state.flatten() == CLOSED)
        info['valid_actions'] = self.valid_actions
        info['num_actions'] = self.num_actions
        return next_state, reward, done, info

    def next_step(self, state, x, y):
        """
        Get the next observation, reward, done, and info.
        Parameters
        ----
        state : (np.array)    visible board
        x : int    location
        y : int    location
        Returns
        ----
        next_state : (np.array)    next visible board
        reward : float               the reward
        done : bool           whether the game end or not
        info : {}
        """
        my_board = state
        if not is_new_move(my_board, x, y):
            return my_board, 0, False, {}
        while True:
            state, game_over = self.get_next_state(my_board, x, y)
            if not game_over:
                #print(my_board)
                #print(state)
                if is_win(state):
                    return state, 100, True, {}
                elif (my_board == state).all():
                    return state, -10, False, {}
                else:
                    return state, 10, False, {}
            else:
                return state, -100, True, {}

    def render(self, mode='human'):
        """
        See gym.Env.render().
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = board2str(self.my_board)
        outfile.write(s)
        if mode != 'human':
            return outfile


# In[4]:


# env 2

import random
import numpy as np
import pandas as pd
from IPython.display import display

class MinesweeperEnv(object):
    def __init__(self, width, height, n_mines,
        # based on https://github.com/jakejhansen/minesweeper_solver
        rewards={'win':1, 'lose':-1, 'progress':0.3, 'guess':-0.3, 'no_progress' : -0.3}):
        self.nrows, self.ncols = width, height
        self.ntiles = self.nrows * self.ncols
        self.n_mines = n_mines
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()
        self.n_clicks = 0
        self.n_progress = 0
        self.n_wins = 0
        self.window = None

        self.rewards = rewards

    def init_grid(self):
        board = np.zeros((self.nrows, self.ncols), dtype='object')
        mines = self.n_mines

        while mines > 0:
            row, col = random.randint(0, self.nrows-1), random.randint(0, self.ncols-1)
            if board[row][col] != 'B':
                board[row][col] = 'B'
                mines -= 1

        return board

    def get_neighbors(self, coord):
        x,y = coord[0], coord[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows)):
                    neighbors.append(self.grid[row,col])

        return np.array(neighbors)

    def count_bombs(self, coord):
        neighbors = self.get_neighbors(coord)
        return np.sum(neighbors=='B')

    def get_board(self):
        board = self.grid.copy()

        coords = []
        for x in range(self.nrows):
            for y in range(self.ncols):
                if self.grid[x,y] != 'B':
                    coords.append((x,y))

        for coord in coords:
            board[coord] = self.count_bombs(coord)

        return board

    def get_state_im(self, state):
        '''
        Gets the numeric image representation state of the board.
        This is what will be the input for the DQN.
        '''

        state_im = [t['value'] for t in state]
        state_im = np.reshape(state_im, (self.nrows, self.ncols, 1)).astype(object)

        state_im[state_im=='U'] = -1
        state_im[state_im=='B'] = -2

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        return state_im

    def init_state(self):
        unsolved_array = np.full((self.nrows, self.ncols), 'U', dtype='object')

        state = []
        for (x, y), value in np.ndenumerate(unsolved_array):
            state.append({'coord': (x, y), 'value':value})

        state_im = self.get_state_im(state)

        return state, state_im

    def color_state(self, value):
        if value == -1:
            color = 'white'
        elif value == 0:
            color = 'slategrey'
        elif value == 1:
            color = 'blue'
        elif value == 2:
            color = 'green'
        elif value == 3:
            color = 'red'
        elif value == 4:
            color = 'midnightblue'
        elif value == 5:
            color = 'brown'
        elif value == 6:
            color = 'aquamarine'
        elif value == 7:
            color = 'black'
        elif value == 8:
            color = 'silver'
        else:
            color = 'magenta'

        return f'color: {color}'

    def draw_state(self, state_im):
        state = state_im * 8
        state_df = pd.DataFrame(state.reshape((self.nrows, self.ncols)), dtype=np.int8)

        display(state_df.style.applymap(self.color_state))

    def render(self, mode):
        if mode == 'human':
            self.draw_state(self.state_im)
        elif mode == 'window':
            state = self.state_im * 8
            self.window = MinesweeeperVisualizer()
            self.window.start(self.nrows, self.ncols, self.n_mines)
            self.window._draw(state)
    
    
    def click(self, action_index):
        coord = self.state[action_index]['coord']
        value = self.board[coord]

        # ensure first move is not a bomb
        if (value == 'B') and (self.n_clicks == 0):
            grid = self.grid.reshape(1, self.ntiles)
            move = np.random.choice(np.nonzero(grid!='B')[1])
            coord = self.state[move]['coord']
            value = self.board[coord]
            self.state[move]['value'] = value
        else:
            # make state equal to board at given coordinates
            self.state[action_index]['value'] = value

        # reveal all neighbors if value is 0
        if value == 0.0:
            self.reveal_neighbors(coord, clicked_tiles=[])

        self.n_clicks += 1

    def reveal_neighbors(self, coord, clicked_tiles):
        processed = clicked_tiles
        state_df = pd.DataFrame(self.state)
        x,y = coord[0], coord[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows) and
                    ((row, col) not in processed)):

                    # prevent redundancy for adjacent zeros
                    processed.append((row,col))

                    index = state_df.index[state_df['coord'] == (row,col)].tolist()[0]

                    self.state[index]['value'] = self.board[row, col]

                    # recursion in case neighbors are also 0
                    if self.board[row, col] == 0.0:
                        self.reveal_neighbors((row, col), clicked_tiles=processed)

    
    def get_action(self):
        board = self.state_im.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

        rand = np.random.random() # random value b/w 0 & 1

        
        move = np.random.choice(unsolved)
        

        return move
    
    def reset(self):
        self.n_clicks = 0
        self.n_progress = 0
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()
        return self.state_im

    def step(self, action_index):
        done = False
        coords = self.state[action_index]['coord']

        current_state = self.state_im

        # get neighbors before action
        neighbors = self.get_neighbors(coords)

        self.click(action_index)

        # update state image
        new_state_im = self.get_state_im(self.state)
        self.state_im = new_state_im

        if self.state[action_index]['value']=='B': # if lose
            reward = self.rewards['lose']
            done = True

        elif np.sum(new_state_im==-0.125) == self.n_mines: # if win
            reward = self.rewards['win']
            done = True
            self.n_progress += 1
            self.n_wins += 1

        elif np.sum(self.state_im == -0.125) == np.sum(current_state == -0.125):
            reward = self.rewards['no_progress']

        else: # if progress
            if all(t==-0.125 for t in neighbors): # if guess (all neighbors are unsolved)
                reward = self.rewards['guess']

            else:
                reward = self.rewards['progress']
                self.n_progress += 1 # track n of non-isoloated clicks

        return self.state_im, reward, done, {} 


# In[5]:


env = MinesweeperEnv(4,4,4)


# In[7]:


#env = MinesweeperDiscreetEnv()
env = MinesweeperEnv(4,4,4)

# RESET
observation = env.reset()



# TEST
#print("Observation space: ", env.get_board())
#print("Shape: ", env.get_board.shape)
#print("Action: ", env.get_action())
#print("Shape: ", env.action_space.shape)

#env.draw_state()
print()
for _ in range(100):
    #print("state: \n", env.state_im)
    env.render('window')
    
    action =  env.get_action()
    state, reward, done, info = env.step(action)

        
    print("Action", action)
    print(f"Reward: {reward} Done: {done}")
    if done:
        print("Game Finished!")
        break
#print("\nObeservation: \n", state)
#env.close()
#env.draw_state(env.state_im)
env.render('window')
env.window.close(True)

# RESET
env.reset()

