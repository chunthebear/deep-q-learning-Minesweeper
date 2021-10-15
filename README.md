# AI Minesweeper with Q Learning and Deep Q Learning
 
## Abstract
Minesweeper is a famous puzzle game involving a single player, re-quiring them to clear a board with hidden mines and numerical cluesindicating the number of mines in the neighbourhood.  We have im-plemented the Q learning and deep Q learning algorithms, ran sev-eral experiments on the reward structures, tuned hyperparameters,trained two final agents, and achieved different levels of success. Boththe  agents  performed  substantially  better  than  a  baseline  randomagent.  Given the limitation of training time,  the Q learning agentperformed better in average reward and board completion rate, butthe deep Q learning agent had a higher winning rate.  The latter islikely to perform better if trained for longer, and is ultimately bettersuited for larger boards with continuous state spaces due to its abilityto predict best actions for unseen states.

## 1 Introduction

Minesweeper is a puzzle game with the objective of clearing a rectangular board of hidden
mines. For each move, the player chooses one cell on the board, revealing either a mine
or information about adjacent cells. The player either wins when all unopened cells con-
tain mines, or loses after revealing a mine. The game of Minesweeper is NP-complete [1],
meaning it is possible to implement a brute-force solver. Despite this possibility, a more
interesting approach to solving the game is using machine learning. Our objective is to
develop Minesweeper solvers using Q-learning and deep Q-learning.

## 2 Problem
The problem is to develop a Q-learning and deep Q-learning agent to play the game
Minesweeper. Due to the nature of reinforcement learning, datasets cannot be used for
this purpose. Instead, a Minesweeper gym environment was developed to train our agent
to allow for the selection of actions.

Minesweeper, as a puzzle game, involves logic, strategies, and pattern recognition. By
training an agent to solve games, we can observe which decisions the agent makes based on
patterns humans may not be able to notice or understand.

### 2.1 Approach
We configured a training environment to simulate Minesweeper games. Then, the Q learn-
ing algorithm is implemented, and the Q-learning agent is trained with different reward
structures. After finding the most suitable structure that provides the best results, hy-
perparameters are tuned over approximately 8 million games. Due to the high number of
possible states, we only trained our agent for a small board size of 4 x 4. To demonstrate the
agent's learning, its performance was then evaluated against the performance of an agent
that made randomly generated moves.

After observing the results of the Q-learning agent, we implemented and trained a deep
Q-learning agent by using the same optimal hyperparameter values and the Deep Rein-
forcement Learning Library (keras-rl2).

### 2.2 Goals

We will measure the success of our agent by its ability to win randomly-generated games.
To measure progression, we will track the win rate, board completion, and average total
reward as the agent is trained, then evaluate the same metrics on the fully-trained agent to
quantify its success.

We also ran an untrained agent to collect the win rates of randomly generated moves; and
compared our results with this. Our goal was to at least beat the scores of this untrained
agent; a larger the difference in the win rates would prove our agent has learnt more.

## 3 Related Work

In our initial research, we discovered several reports describing various approaches to solving
Minesweeper. Gardea, Koontz, and Silva achieved up to a win rate of 32% using a combi-
nation of logistic regression, support-vector machines, and simplified Q-learning [2]. Most
inspirationally, Stephen Lee outlined his approach to configuring a deep Q-learning agent
that reached an 80% win rate after training [3]. Previous attempts using simple Q-Learning
have failed to achieve win rates of greater than 5% with board sizes upwards of 6 x 6 [4],
which factored into our decision to limit the problem to a 4 x 4 board.

## 4 Environment

Because of the nature of the problem, we cannot use pre-existing datasets to train the
agent. Rather, we configured a training environment to simulate Minesweeper games. In
the environment,

- a game represents an episode,
- the board represents the state, and
- each possible move represents an action.

The first environment we used was a naive implementation, containing a basic 2D matrix
representing the game board. Each matrix value contained one of: a number from 0 to
8, corresponding to the number of adjacent mines; a-2, indicating an unopened cell; or
􀀀1, a mine. The scope of possible actions is any cell { imposing the possibility of selecting
a previously-opened cell. Because of this flaw, and the lack of penalization for repeated
actions, the agent would often get stuck and fail to learn.

A more advanced environment was implemented, restricting valid actions to only unopened
cells. The new environment also opens neighbouring cells when a safe cell is selected, in
accordance with the real Minesweeper game. These changes drastically reduced the number
of steps taken to complete a game, making training both more efficient and more effective.
Additionally, in the new environment, if the first opened cell reveals a mine, the game will
silently restart to avoid failing on the first move and not learning. This advanced environ-
ment was configured for use with both Q-learning and deep Q-learning.

### 4.1 Reward Structure

The initial reward structure (Structure 1) was brainstormed. Initially, the positively re-
warded actions were winning a game, and opening a new cell and losing the game was a
negative reward. When using such a simple reward structure with our very first environment
(the most naive Gym environment), we found that even after 1000 runs, the agent failed
to learn to not click on an already opened cell. This made sense as the reward structure
was flawed, since reopening a cell would still result in the agent being positively rewarded.
Hence, in our next version of the reward structure (Structure 2), we gave a negative reward
to the agent for clicking on the already opened cell, and kept everything else the same as
Structure 1.

A more advanced reward structure (Structure 3) was then adapted to have the agent re-
warded positively for winning and opening a new cell (strategically, based on what the agent
has learnt so far). Structure 3 rewarded the agent negatively for guessing which cell to open,
reopening an open cell, and losing.

We did some research by actually playing the Minesweeper game, and observing someone
who has never played the game to devise reward strategies that would help an agent learn
more effectively. Based on that, we designed a new reward structure (Structure 4). Here,
the agent was rewarded like in Structure 2, except in Structure 2 the reward for opening a
new cell was a constant whereas in Structure 4, we decided to take a dynamic approach for
opening a new cell. The new reward for opening a new cell was based on the percentage of
new cells opened by the action - an action that chose a cell adjacent to a mine would be
rewarded less than an action that would open neighbouring cells.

Table 1 states the reward values for each action. The values are consistent throughout
all four reward structures, except for the opening a cell action in Structure 4.

| Action      | Reward |
| ----------- | ----------- |
|Opening a new cell | +0.3|
|Reopening a cell | -0.3|
|Guessing a cell | -0.3|
|Victory | +1.0|
|Loss | -1.0|

Table 1: Reward summary

#### 4.1.1 Comparing Reward Structures

We excluded Structure 1 from our tests, since we had already established the poor perfor-
mance of the agent using that structure.

To compare the three remaining structures, we tested multiple episodes on different board
sizes (Appendix A). The performance is judged based on the percentage of games won dur-
ing testing. The overall results pointed towards the comparatively worse performance of
Structure 4. However, the performance of Structure 2 and Structure 3 were relatively close.
To further obtain the best structure to use, we compared Structure 2 and Structure 3.
This time, we used a bigger board and more episodes for a deeper comparison. The results,
as seen in the figures below, manifest that the agent trained with Structure 2 (which is one
step simpler than Structure 3) was able to win a higher percentage of games consistently,
even though not by a substantial difference. This could potentially mean that Structure 3
is overcomplicating the reward structure, and trying harder to control the agent does not
lead to better results.

![Figure 1](/pics/f1.png)

Since Structure 2 performed better when tested, we chose to train our Q-learning and deep
Q-learning agents using this structure. Recall that this structure positively rewarded for
winning (+1) and opening a new cell (+0.3), and negatively rewarded for losing (-1) and
reopening an already opened cell (-0.3).

## 5 Q-Learning
The implementation of Q-learning involves the construction of a Q-table which holds a Q-
value for each pairwise combination of states and actions. For any state, the action that
provides the maximum Q-value for that state in the table represents the best possible action.
After each action, the corresponding Q-value is updated by using the Bellman equation [5]:

![Bellman](/pics/bell.png)

The agent learns through a combination of exploitation and exploration: choosing the best
possible action using the Q-table, or choosing a random action, respectively. During training,
the decision between exploration and exploitation is dictated by the exploration parameter
epsilon, representing the chance an action is explorative. At each step, the exploration rate is
reduced to progressively reduce the amount of random actions taken. The basic pseudocode
for the algorithm is defined below [6].

![image](https://user-images.githubusercontent.com/16961563/137453865-98a56986-313b-4cf0-afd8-a65b3ecd083f.png)
