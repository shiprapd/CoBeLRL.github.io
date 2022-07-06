# basic imports
import numpy as np
from scipy import linalg


class DynaQMemory():
    """
    Memory module to be used with the Dyna-Q agent.
    Experiences are stored as a static table.
    """
    
    def __init__(self, numberOfStates, numberOfActions, learningRate=0.9):
        """
        Memory module to be used with the Dyna-Q agent.
        Experiences are stored as a static table.

        :param numberOfStates:               The number of environmental states
        :param numberOfActions:              The number of the agent's actions
        :param learningRate:                 The learning rate with which experiences are updated
        :return: None
        """
        # initialize variables
        self.learningRate = learningRate
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
        self.rewards = np.zeros((numberOfStates, numberOfActions))
        self.states = np.tile(np.arange(self.numberOfStates).reshape(self.numberOfStates, 1), self.numberOfActions).astype(int)
        self.terminals = np.zeros((numberOfStates, numberOfActions)).astype(int)
        


    def store(self, experience):
        """
        This function stores a given experience.
        
        :param experience:    The experience to be stored.
        :return: None
        """
        
        # update experience
        self.rewards[experience['state']][experience['action']] += self.learningRate * (experience['reward'] - self.rewards[experience['state']][experience['action']])
        self.states[experience['state']][experience['action']] = experience['next_state']
        self.terminals[experience['state']][experience['action']] = experience['terminal']



    def retrieve(self, state, action):
        """
        This function retrieves a specific experience.
        
        :param state: The environmental state.
        :param action: The action selected.
        :return (state, action, reward, next_state, terminal: A dictionary containing the experience
        """

        return {'state': state, 'action': action, 'reward': self.rewards[state][action],
                'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}
        


    def retrieve_batch(self, numberOfExperiences=1):
        """
        This function retrieves a number of random experiences.
        
        :param numberOfExperiences: The number of random experiences to be drawn
        :return experiences: Random experiences retrieved
        """

        # Draw random experiences
        idx = np.random.randint(0, self.numberOfStates * self.numberOfActions, numberOfExperiences)
        # Determine indeces
        idx = np.array(np.unravel_index(idx, (self.numberOfStates, self.numberOfActions)))
        # Build experience batch
        experiences = []
        for exp in range(numberOfExperiences):
            state, action = idx[0, exp], idx[1, exp]
            experiences += [{'state': state, 'action': action, 'reward': self.rewards[state][action],
                             'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}]
            
        return experiences
    



class PMAMemory():
    """
    Class for Memory module to be used with the Dyna-Q agent.
    Experiences are stored as a static table.
    """
    
    def __init__(self, interfaceOAI, rlAgent, numberOfStates, numberOfActions, learningRate=0.9, gamma=0.9):
        """
        Inits Memory module to be used with the Dyna-Q agent.
        Experiences are stored as a static table.

        :param numberOfStates: The number of environmental states
        :param numberOfActions: The number of the agent's actions
        :param learningRate:  The learning rate with which experiences are updated
        :return: None
        """
        # Initialize variables
        self.learningRate = learningRate
        self.learningRateT = 0.9
        self.gamma = gamma
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
        self.minGain = 10 ** -6
        self.minGainMode = 'original'
        self.equal_need = False
        self.equal_gain = False
        self.ignore_barriers = True
        self.allow_loops = False
        # Initialize memory structures
        self.rewards = np.zeros((numberOfStates, numberOfActions))
        self.states = np.zeros((numberOfStates, numberOfActions)).astype(int)
        self.terminals = np.zeros((numberOfStates, numberOfActions)).astype(int)
        # Store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI
        # Store reference to agent
        self.rlParent = rlAgent
        # Compute state-state transition matrix
        self.T = np.sum(self.interfaceOAI.world['sas'], axis=1)/self.interfaceOAI.action_space.n
        # Compute successor representation
        self.SR = np.linalg.inv(np.eye(self.T.shape[0]) - self.gamma * self.T)
        # Determine transitions that should be ignored (i.e. those that lead into the same state)
        self.update_mask = (self.states.flatten(order='F') != np.tile(np.arange(self.numberOfStates), self.numberOfActions))



    def store(self, experience):
        """
        This function stores a given experience.
        
        :param experience:     The experience to be stored.
        :return: None
        """
        # Update experience
        self.rewards[experience['state']][experience['action']] += self.learningRate * (experience['reward'] - self.rewards[experience['state']][experience['action']])
        self.states[experience['state']][experience['action']] = experience['next_state']
        self.terminals[experience['state']][experience['action']] = experience['terminal']
        # Update T
        self.T[experience['state']] += self.learningRateT * ((np.arange(self.numberOfStates) == experience['next_state']) - self.T[experience['state']])


    def replay(self, replayLength, current_state, force_first=None):
        """
        This function replays experiences.
        
        :param replayLength:  The number of experiences that will be replayed
        :param current_state: State at which replay should start
        :return performed_updates: list of performed updates
        """
        
        performed_updates = []
        last_seq = 0
        for update in range(replayLength):
            # Make list of 1-step backups
            updates = []
            for i in range(self.numberOfStates * self.numberOfActions):
                s, a = i % self.numberOfStates, int(i/self.numberOfStates)
                updates += [[{'state': s, 'action': a, 'reward': self.rewards[s, a], 'next_state': self.states[s, a], 'terminal': self.terminals[s, a]}]]
            # Extend current update sequence
            extend = -1
            if len(performed_updates) > 0:
                # Determine extending state
                extend = performed_updates[-1]['next_state']
                # Check for loop
                loop = False
                for step in performed_updates[last_seq:]:
                    if extend == step['state']:
                        loop = True
                # Extend update
                if not loop or self.allow_loops:
                    # Determine extending action which yields max future value
                    extending_action = self.actionProbs(self.rlParent.Q[extend])
                    extending_action = np.random.choice(np.arange(self.numberOfActions), p=extending_action)
                    # Determine extending step
                    extend += extending_action * self.numberOfStates
                    # Updates[extend] = performed_updates[last_seq:] + updates[extend]
                    updates[extend] = performed_updates[-1:] + updates[extend]
            # Compute gain and need
            gain = self.computeGain(updates)
            if self.equal_gain:
                gain.fill(1)
            need = self.computeNeed(current_state)
            if self.equal_need:
                need.fill(1)
            # Determine backup with highest utility
            utility = gain * need
            if self.ignore_barriers:
                utility *= self.update_mask
            # Determine backup with highest utility
            ties = (utility == np.amax(utility))
            utility_max = np.random.choice(np.arange(self.numberOfStates * self.numberOfActions), p=ties/np.sum(ties))
            # Force initial update
            if len(performed_updates) == 0 and force_first is not None:
                utility_max = force_first +  np.random.randint(self.numberOfActions) * self.numberOfStates
            # Perform update
            self.rlParent.update_Q(updates[utility_max])
            # Add update to list
            performed_updates += [updates[utility_max][-1]]
            if extend != utility_max:
                last_seq = update
            
        return performed_updates
            


    def computeGain(self, updates):
        """
        This function computes the gain for each possible n-step backup in updates.
        
        :param updates:      A list of n-step updates.
        :return gains:        Gain for each possible n-step in updates
        """

        gains = []
        for update in updates:
            gain = 0.
            # Expected future value
            future_value = np.amax(self.rlParent.Q[update[-1]['next_state']]) * update[-1]['terminal']
            # Gain is accumulated over the whole trajectory
            for s, step in enumerate(update):
                # Policy before update
                policy_before = self.actionProbs(self.rlParent.Q[step['state']])
                # Sum rewards over subsequent n-steps
                R = 0.
                for following_steps in range(len(update) - s):
                    R += update[s + following_steps]['reward'] * (self.gamma ** following_steps)
                # Compute new Q-value
                q_target = np.copy(self.rlParent.Q[step['state']])
                q_target[step['action']] = R + future_value * (self.rlParent.gamma ** (following_steps + 1))
                q_new = self.rlParent.Q[step['state']] + self.rlParent.learningRate * (q_target - self.rlParent.Q[step['state']])
                # Policy after update
                policy_after = self.actionProbs(q_new)
                # Compute gain
                step_gain = np.sum(q_new * policy_after) - np.sum(q_new * policy_before)
                if self.minGainMode == 'original':
                    step_gain = max(step_gain, self.minGain)
                # Add gain
                gain += step_gain
            # Store gain for current update
            gains += [max(gain, self.minGain)]
        
        return np.array(gains)
    


    def computeNeed(self, currentState=None):
        """
        This function computes the need for each possible n-step backup in updates.
        
        :param currentState: The state that the agent currently is in
        :return: None
        """

        # Use standing distribution of the MDP for 'offline' replay
        if currentState is None:
            # Compute left eigenvectors
            eig, vec = linalg.eig(self.T, left=True, right=False)
            best = np.argmin(np.abs(eig - 1))
            
            return np.tile(np.abs(vec[:,best].T), self.numberOfActions)
        # Use SR given the current state for 'awake' replay
        else:
            return np.tile(self.SR[currentState], self.numberOfActions)
    


    def updateSR(self):
        """
        This function updates the SR given the current state-state transition matrix T. 
        
        :param:  None
        :return: None
        """

        self.SR = np.linalg.inv(np.eye(self.T.shape[0]) - self.gamma * self.T)
        


    def updateMask(self):
        """
        This function updates the update mask. 
        
        :param:  None
        :return: None
        """
        
        self.update_mask = (self.states.flatten(order='F') != np.tile(np.arange(self.numberOfStates), self.numberOfActions))
    


    def actionProbs(self, q):
        """
        This function computes the action probabilities given a set of Q-values.
        
        :param q:      The set of Q-values.
        :return p:      Action Probabilities
        """

        # Assume greedy policy per default
        ties = (q == np.amax(q))
        # p = ties/np.sum(ties)
        p = np.ones(self.numberOfActions) * (self.rlParent.epsilon/self.numberOfActions)
        p[ties] += (1. - self.rlParent.epsilon)/np.sum(ties)
        # p = np.arange(self.numberOfActions) == np.argmax(q)
        # Softmax when 'on-policy'
        if self.rlParent.policy == 'softmax':
            # Catch all zero case
            if np.all(q == q[0]):
                q.fill(1)
            p = np.exp(q * self.rlParent.beta)
            p /= np.sum(p)
            
        return p
