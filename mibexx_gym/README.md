# [GYM] Minesweeper Environment

A simple minesweeper environment for gym.

## Goal
Goal is to reveal all fields without a mine.

## Install

### PIP
```
pip install mibexx-gym-minesweeper
```

### Source
```
pip install -e .
```

## Initialize
```python
import gym

env = gym.make('mibexx_gym_minesweeper:mibexx-gym-minesweeper-v0')
env.reset()
```


## Environment

### State
Observation space:  (9, 9)  
Low: -2  
High: 8

```
[[0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]]
```

#### Values
| Value | Description |
| ----- | ----------- |
| -2 | Mine |
| -1 | Empty field, no mine near |
| 0 | Not revealed |
| 1 | Revealed, 1 mine near |
| 2 | Revealed, 2 mines near |
| 3 | Revealed, 3 mines near |
| 4 | Revealed, 4 mines near |
| 5 | Revealed, 5 mines near |
| 6 | Revealed, 6 mines near |
| 7 | Revealed, 7 mines near |
| 8 | Revealed, 8 mines near |

### Actions
Action space:  [9 9]

Action is a (y,x) tuple.

Example for row 3 col 2:
```python
state, reward, done, _ = env.step((3, 2))
```


## Rewards
empty field: 1  
already clicked field: -1   
a mine: -10  