# basic imports
import numpy as np
import pyqtgraph as pg


class RLPerformanceMonitorBaseline():
    """
    Performance monitor. Used for tracking learning progress.
    """

    def __init__(self, rlAgent, trials, guiParent, visualOutput):
        """
        Inits Performance monitor class. Used for tracking learning progress.

        :param rlAgent: Reference to the RL agent used.
        :param trials: Maximum number of trials for which the experiment is run.
        :param guiParent: The main window for visualization.
        :param visualOutput: If true, the learning progress will be plotted.
        :return: None
        """
        # store the rlAgent
        self.rlAgent = rlAgent
        self.guiParent = guiParent
        # shall visual output be provided?
        self.visualOutput = visualOutput
        # define the variables that will be monitored
        self.rlRewardTraceRaw = np.zeros(trials, dtype='float')
        self.rlRewardTraceRefined = np.zeros(trials, dtype='float')
        # this is the accumulation range for smoothing the reward curve
        self.accumulationRangeReward = 20
        # this is the accumulation interval for correct/incorrect decisions at the beginning/end of the single
        # experimental phases (acquisition,extinction,renewal)
        self.accumulationIntervalPerformance = 10

        if visualOutput:
            # redefine the gui's dimensions
            self.guiParent.setGeometry(50, 50, 1600, 600)
            # set up the required plots
            self.rlRewardPlot = self.guiParent.addPlot(title="Reinforcement learning progress")
            # set x/y-ranges for the plots
            self.rlRewardPlot.setXRange(0, trials)
            self.rlRewardPlot.setYRange(-100., 100.)
            # define the episodes domain
            self.episodesDomain = np.linspace(0, trials, trials)
            # each variable has a dedicated graph that can be used for displaying the monitored values
            self.rlRewardTraceRawGraph = self.rlRewardPlot.plot(self.episodesDomain, self.rlRewardTraceRaw)
            self.rlRewardTraceRefinedGraph = self.rlRewardPlot.plot(self.episodesDomain, self.rlRewardTraceRefined)

    def clearPlots(self):
        """
        This function clears the plots generated by the performance monitor.

        :param:None
        :return:None
        """
        if self.visualOutput:
            self.guiParent.removeItem(self.rlRewardPlot)

    def update(self, trial, logs):
        """
        This function is called when a trial ends. Here, information about the monitored variables is memorized,
        and the monitor graphs are updated.

        :param trial: The actual trial number
        :param logs: Information from the reinforcement learning subsystem
        :return: None

        """
        # update the reward traces
        rlReward = logs['episode_reward']
        self.rlRewardTraceRaw[trial] = rlReward
        # prepare aggregated reward trace
        aggregatedRewardTraceRaw = None
        if trial < self.accumulationRangeReward:
            aggregatedRewardTraceRaw = self.rlRewardTraceRaw[trial:None:-1]
        else:
            aggregatedRewardTraceRaw = self.rlRewardTraceRaw[trial:trial - self.accumulationRangeReward:-1]
        self.rlRewardTraceRefined[trial] = np.mean(aggregatedRewardTraceRaw)

        if self.visualOutput:
            # set the graph's data
            self.rlRewardTraceRawGraph.setData(self.episodesDomain, self.rlRewardTraceRaw,
                                               pen=pg.mkPen(color=(128, 128, 128), width=1))
            self.rlRewardTraceRefinedGraph.setData(self.episodesDomain, self.rlRewardTraceRefined,
                                                   pen=pg.mkPen(color=(255, 0, 0), width=2))
