import numpy as np
import json
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


class DQNAgentBaseline():
    """
    This is a reinforcement learning class. It wraps all functionality required to set up a RL agent.
    This is the constructor method.  The sequential model is developed and
    the agent is constructed.
    """

    class callbacks(callbacks.Callback):
        """
        This nested visualization class is required by 'KERAS-RL' to visualize the training success
        (by means of episode reward) at the end of each episode, and update the policy visualization.
        """

        def __init__(self, rlParent, trialBeginFcn=None, trialEndFcn=None):
            """
            Inits nested visualization class is required by 'KERAS-RL' to visualize the training success
            (by means of episode reward) at the end of each episode, and update the policy visualization.

            :param rlParent: The ACT_ReinforcementLearningModule that hosts this class
            :param trialBeginFcn: The callback function called in the beginning of each trial, defined for more flexibility in scenario control
            :param trialEndFcn: The callback function called at the end of each trial, defined for more flexibility in scenario control
            :return: None
            """

            super(DQNAgentBaseline.callbacks, self).__init__()

            # Store the hosting class
            self.rlParent = rlParent

            # Store the trial end callback function
            self.trialBeginFcn = trialBeginFcn
            self.trialEndFcn = trialEndFcn

        def on_episode_begin(self, epoch, logs):
            """
            The function is called whenever an episode starts,
            and updates the visual output in the plotted reward graphs.

            :param epoch: The number of passes of the dataset
            :param logs: Log data
            :return: None
            """
            # Retrieve the Open AI Gym interface
            interfaceOAI = self.rlParent.interfaceOAI

            if self.trialBeginFcn is not None:
                self.trialBeginFcn(epoch, self.rlParent)

        def on_episode_end(self, epoch, logs):
            """
            The function is called whenever an episode ends, and updates the reward accumulator,
            simultaneously updating the visualization of the reward function.

            :param epoch: The number of passes of the dataset
            :param logs: Log data
            :return: None
            """

            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)

    def __init__(self, interfaceOAI, memoryCapacity=10000, epsilon=0.3, processor=None,
                 trialBeginFcn=None, trialEndFcn=None, lr=0.001, network=None):
        """
        Inits a reinforcement learning class. It wraps all functionality required to set up a RL agent.

        :param interfaceOAI:      The interface to the Open AI Gym environment
        :param memoryCapacity:    The capacity of the sequential memory used in the agent
        :param epsilon:           The epsilon value for the epsilon greedy policy
        :param processor:         Abstract base class for implementing processor
        :param trialBeginFcn:     The callback function called at the beginning of each trial, defined for more flexibility in scenario control
        :param trialEndFcn:       The callback function called at the end of each trial, defined for more flexibility in scenario control
        :param lr:                The learning rate with which the Q-function is updated.
        :param network:           The DNN to be used by the agent. If None, a dense DNN is created by default.
        :return:  None
        """

        # Store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI

        # Prepare the model used in the reinforcement learner #

        # The number of discrete actions, retrieved from the Open AI Gym interface
        self.nb_actions = self.interfaceOAI.action_space.n

        # A sequential model is standard used here, this model is subject to changes
        if network is None:
            self.model = Sequential()
            self.model.add(Flatten(input_shape=(1,) + self.interfaceOAI.observation_space.shape))
            self.model.add(Dense(units=64, activation='tanh'))
            self.model.add(Dense(units=64, activation='tanh'))
            self.model.add(Dense(units=64, activation='tanh'))
            self.model.add(Dense(units=64, activation='tanh'))
            self.model.add(Dense(units=self.nb_actions, activation='linear'))

        else:
            loaded_model_json = json.dumps(network)
            self.model = model_from_json(loaded_model_json)
            print(self.model.summary())

        # Prepare the memory for the RL agent
        self.memory = SequentialMemory(limit=memoryCapacity, window_length=1)

        # Define the available policies
        policyEpsGreedy = EpsGreedyQPolicy(epsilon)

        ##### Construct the agent #####

        # Retrieve the agent's parameters from the agentParams dictionary
        self.agent = DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=self.memory, nb_steps_warmup=100,
                              enable_dueling_network=False, dueling_type='avg', target_model_update=1e-2,
                              policy=policyEpsGreedy, batch_size=32, processor=processor)

        # Compile the agent
        self.agent.compile(Adam(lr=lr, ), metrics=['mse'])

        # Set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self, trialBeginFcn, trialEndFcn)

    def train(self, steps):
        """
        This method is called to train the agent using the fit method.

        :param steps: Maximum number of steps required to train the model.
        :return: None
        """
        # Call the fit method to start the RL learning process
        self.maxSteps = steps
        self.agent.fit(self.interfaceOAI, nb_steps=steps, verbose=0, callbacks=[self.engagedCallbacks],
                       nb_max_episode_steps=100, visualize=False)
