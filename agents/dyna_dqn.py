# Basic imports
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Framework imports
from agents.dyna_q_agent import AbstractDynaQAgent
from memory_modules.dyna_q_memory import DynaQMemory


class DynaDQN(AbstractDynaQAgent):
    """
    Implementation of a DQN agent using the Dyna-Q model.
    This agent uses the Dyna-Q agent's memory module and then maps gridworld states to predefined observations.
    """

    class callbacks():
        """
        Callback class. Used for visualization and scenario control.
        """
        def __init__(self, rlParent, trialEndFcn=None):
            """
            Inits callback class

            :param rlParent:       Reference to the Dyna-Q agent.
            :param trialEndFcn:    Maximum number of experiences that will be stored by the memory module.
            :return: None
            """

            super(DynaDQN.callbacks, self).__init__()
            self.rlParent = rlParent  # Store the hosting class
            self.trialEndFcn = trialEndFcn  # Store the trial end callback function

        def on_episode_end(self, epoch, logs):
            """
            This function is called whenever an episode ends, and updates the reward accumulator,
            simultaneously updating the visualization of the reward function.

            :param epoch:  The number of passes of the dataset
            :param logs:   log data
            :return: None
            """

            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)

    def __init__(self, interfaceOAI, epsilon=0.3, beta=5, learningRate=0.9, gamma=0.99, trialEndFcn=None,
                 observations=None, model=None):
        """
        Inits DQN agent using the Dyna-Q model

        :param interfaceOAI:   The interface to the Open AI Gym environment.
        :param epsilon:        The epsilon value for the epsilon greedy policy.
        :param beta:           The temperature parameter used when applying the softmax function to the Q-values.
        :param learningRate:   The learning rate with which the Q-function is updated.
        :param gamma:          The discount factor used to compute the TD-error.
        :param trialEndFcn:    Callback function called at the end of each trial,
                                defined for flexibility in scenario control.
        :param observations:   The set of observations that will be mapped to the gridworld states.
        :param model:          The DNN model to be used by the agent.
        :return: None
        """
        super().__init__(interfaceOAI, epsilon=epsilon, beta=beta, learningRate=learningRate, gamma=gamma)

        # Prepare observations
        if observations is None or observations.shape[0] != self.numberOfStates:
            self.observations = np.eye(self.numberOfStates)  # One-hot encoding of states
        else:
            self.observations = observations
        # Build target and online models
        self.build_model(model)
        self.current_predictions = self.model_online.predict_on_batch(self.observations)
        self.M = DynaQMemory(self.interfaceOAI.world['states'], self.numberOfActions)  # Memory module
        # Set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self, trialEndFcn)
        # Perform replay at the end of an episode instead of each step
        self.episodic_replay = False
        # The rate at which the target model is updated (for values< 1, target model is blended with the online model)
        self.target_model_update = 10 ** -2
        # Count the steps since the last update of the target model
        self.steps_since_last_update = 0

    def build_model(self, model=None):
        """
        This function builds the DQN's target and online models.

        :param model: The DNN model to be used by the agent. If None, a small dense DNN is created by default.
        :return: None
        """
        # Build target model
        if model is None:
            self.model_target = Sequential()
            self.model_target.add(Dense(units=64, input_shape=(self.observations.shape[1],), activation='tanh'))
            self.model_target.add(Dense(units=64, activation='tanh'))
            self.model_target.add(Dense(units=self.numberOfActions, activation='linear'))
        else:
            self.model_target = model
        self.model_target.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        # Build online model by cloning the target model
        self.model_online = clone_model(self.model_target)
        self.model_online.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    def train(self, numberOfTrials=100, maxNumberOfSteps=50, replayBatchSize=100, noReplay=False):
        """
        This function is called to train the agent.

        :param numberOfTrials:               The number of trials the Dyna-Q agent is trained.
        :param maxNumberOfSteps:             The maximum number of steps per trial.
        :param replayBatchSize:              The number of random that will be replayed.
        :param noReplay:                     If true, experiences are not replayed.
        :return: None
        """
        for trial in range(numberOfTrials):
            state = self.interfaceOAI.reset()  # Reset environment
            logs = {'episode_reward': 0}  # Log cumulative reward
            for step in range(maxNumberOfSteps):
                self.steps_since_last_update += 1
                action = self.select_action(state, self.epsilon, self.beta)  # Determine next action
                next_state, reward, stopEpisode, callbackValue = self.interfaceOAI.step(action)  # Perform action
                # Make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state,
                              'terminal': (1 - stopEpisode)}
                self.M.store(experience)  # Store experience
                state = next_state  # Update current state
                # Perform experience replay
                if not noReplay and not self.episodic_replay:
                    self.replay(replayBatchSize)
                logs['episode_reward'] += reward  # Update cumulative reward
                # Stop trial when the terminal state is reached
                if stopEpisode:
                    break
            # To save performance store state predictions after each trial only
            self.current_predictions = self.model_online.predict_on_batch(self.observations)
            # Perform experience replay
            if not noReplay and self.episodic_replay:
                self.replay(replayBatchSize)
            # Callback
            self.engagedCallbacks.on_episode_end(trial, logs)

    def replay(self, replayBatchSize=200):
        """
        This function replays experiences to update the Q-function.
        
        :param replayBatchSize:  The number of random that will be replayed.
        :return: None
        """
        # Sample random batch of experiences
        replayBatch = self.M.retrieve_batch(replayBatchSize)
        # Compute update targets
        states, next_states = np.zeros((replayBatchSize, self.observations.shape[1])), np.zeros(
            (replayBatchSize, self.observations.shape[1]))
        rewards, terminals, actions = np.zeros(replayBatchSize), np.zeros(replayBatchSize), np.zeros(replayBatchSize)
        for e, experience in enumerate(replayBatch):
            states[e] = self.observations[experience['state']]
            next_states[e] = self.observations[experience['next_state']]
            rewards[e] = experience['reward']
            actions[e] = experience['action']
            terminals[e] = experience['terminal']
        future_values = np.amax(self.model_target.predict_on_batch(next_states), axis=1) * terminals
        targets = self.model_target.predict_on_batch(states)
        for a, action in enumerate(actions):
            targets[a, int(action)] = rewards[a] + self.gamma * future_values[a]
        # Update online model
        self.model_online.train_on_batch(np.array(states), np.array(targets))
        # Update target model
        if self.target_model_update < 1.:
            weights_target = self.model_target.get_weights()
            weights_online = self.model_online.get_weights()
            for layer in range(len(weights_target)):
                weights_target[layer] += self.target_model_update * (weights_online[layer] - weights_target[layer])
            self.model_target.set_weights(weights_target)
            self.steps_since_last_update = 0
        elif self.steps_since_last_update >= self.target_model_update:
            self.model_target.set_weights(self.model_online.get_weights())
            self.steps_since_last_update = 0

    def update_Q(self, experience):
        """
        This function is a dummy function and does nothing (implementation required by parent class).
        
        :param experience: The experience with which the Q-function will be updated.
        :return: None
        """
        pass

    def retrieve_Q(self, state):
        """
        This function retrieves Q-values for a given state.
        
        :param state: The state for which Q-values should be retrieved.
        :return: Q-values
        """
        return self.model_online.predict_on_batch(np.array([self.observations[state]]))[0]

    def predict_on_batch(self, batch):
        """
        This function retrieves Q-values for a batch of states.
        
        :param batch: The batch of states.
        :returns: Q-values on batch of states
        """
        return self.current_predictions[batch]
