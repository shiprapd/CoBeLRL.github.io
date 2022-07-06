# Basic imports
import numpy as np


def makeGridworld(height, width, terminals=[], rewards=None, goals=[], startingStates=[], invalidStates=[], invalidTransitions=[], wind=None, deterministic=True):
    """
    This function builds a gridworld according to the given parameters.
    
    :param height:                       The gridworld's height.
    :param width:                        The gridworld's width.
    :param terminals:                    The gridworld's terminal states as a list.
    :param rewards:                      The gridworld's state rewards as an array where the first column contains the state indeces and the second column the state rewards.
    :param goals:                        The gridworld's goal states as a list (Used for visualization).
    :param startingStates:               Possible starting states as a list.
    :param invalidStates:                The gridworld's unreachable states as list.
    :param invalidTransitions:           The gridworld's invalid transitions as a list of 2-tuples.
    :param wind:                         The wind applied to the gridworld's states where the first column contains the state indeces and the second and thirds column the wind applied to height and width coordinates.
    :param deterministic:                If true, state transition with the highest probability are chosen.
    :return world:                       Gridworld according to the given parameters
    """
    world = dict()
    # World dimensions as integers
    world['height'] = height
    world['width'] = width
    # Number of world states N
    world['states'] = height * width
    # Goals for visualization as list
    world['goals'] = goals
    # Terminals as arry of size N
    world['terminals'] = np.zeros(world['states'])
    if not terminals is None:
        world['terminals'][terminals] = 1
    # Rewards as array of size N
    world['rewards'] = np.zeros(world['states'])
    if not rewards is None:
        world['rewards'][rewards[:, 0].astype(int)] = rewards[:, 1]
    # Starting states as array of size S
    # If starting states were not defined, all states except the terminals become starting states
    world['startingStates'] = list(set([i for i in range(world['states'])]) - set(terminals))
    if len(startingStates) > 0:
        world['startingStates'] = startingStates
    world['startingStates'] = np.array(world['startingStates'])
    # Wind applied at each state as array of size Nx2
    world['wind'] = np.zeros((world['states'], 2))
    if not wind is None:
        world['wind'][wind[:, 0].astype(int)] = wind[:, 1:]
    # Invalid states and transitions as lists
    world['invalidStates'] = invalidStates
    world['invalidTransitions'] = invalidTransitions
    # State coordinates as array of size Nx2
    world['coordinates'] = np.zeros((world['states'], 2))
    for i in range(width):
        for j in range(height):
            state = j * width + i
            world['coordinates'][state] = np.array([i, height - 1 - j])
    # State-action-state transitions as array of size Nx4xN
    world['sas'] = np.zeros((world['states'], 4, world['states']))
    for state in range(world['states']):
        for action in range(4):
            h = int(state/world['width'])
            w = state - h * world['width']
            # Left
            if action == 0:
                w = max(0, w-1)
            # Up
            elif action == 1:
                h = max(0, h-1)
            # Right
            elif  action == 2:
                w = min(world['width'] - 1, w+1)
            # Down
            else:
                h = min(world['height'] - 1, h+1)
            # Apply wind
            # Currently walls are not taken into account!
            h += world['wind'][state][0]
            w += world['wind'][state][1]
            h = min(max(0, h), world['height'] - 1)
            w = min(max(0, w), world['width'] - 1)
            # Determine next state
            nextState = int(h * world['width'] + w)
            if nextState in world['invalidStates'] or (state, nextState) in world['invalidTransitions']:
                nextState = state
            world['sas'][state][action][nextState] = 1
            
    world['deterministic'] = deterministic
    
    return world
    


def makeOpenField(height, width, goalState=0, reward=1):
    """
    This function builds an open field gridworld with one terminal goal state.
    
    :param:
        height:             The gridworld's height.
        width:              The gridworld's width.
        goalState:          The gridworld's goal state.
        reward:             The reward received upon reaching the gridworld's goal state.
    :return:                Open field gridworld with one terminal goal state.
    """
    return makeGridworld(height, width, terminals=[goalState], rewards=np.array([[goalState, reward]]), goals=[goalState])



def makeEmptyField(height, width):
    """
    This function builds an empty open field gridworld.
    
    :param:
        height:         The gridworld's height.
        width:          The gridworld's width.
    :return:            Empty open field gridworld.
    """
    return makeGridworld(height, width)



def makeWindyGridworld(height, width, columns, goalState=0, reward=1, direction='up'):
    """
    This function builds a windy gridworld with one terminal goal state.
    
    :param height:             The gridworld's height.
    :param width:              The gridworld's width.
    :param columns:            Wind strengths for the different columns.
    :param goalState:          The gridworld's goal state.
    :param reward:             The reward received upon reaching the gridworld's goal state.
    :param direction:          The wind's direction (up, down).
    :return:                Windy gridworld
    """
    directions = {'up': 1, 'down': -1}
    wind = np.zeros((height * width, 3))
    for i in range(width):
        for j in range(height):
            state = int(j * width + i)
            wind[state] = np.array([state, columns[i] * directions[direction], 0])
    return makeGridworld(height, width, terminals=[goalState], rewards=np.array([[goalState, reward]]), goals=[goalState], wind=wind)



if __name__ == '__main__':
    height, width = 5, 5
    goal = 4
    reward = 1
    columns = np.array([0, 0, 0, 1, 1])
    gridworld = makeGridworld(height, width, terminals=[goal], rewards=np.array([[goal, reward]]))
    openField = makeOpenField(height, width, goal, reward)
    windyGridworld = makeWindyGridworld(height, width, columns, goal, reward)
