from .cog_arrow import CogArrow

class TopologyNode():
    """
     This class defines a single node of the topology graph
    """

    def __init__(self,index,x,y):
        """"
        Set up a single node of the topology graph.

        :param x: x position of the node
        :param y: y position of the node
        :param index: The global index of the node
        :return: None
        """

        self.index=index    # The node's global index
        
        # The node's global position
        self.x=x
        self.y=y
        
        # Is this node the requested goal node?
        self.goalNode=False
        
        # The clique of the node's neighboring nodes
        self.neighbors=[]
        
        # An indicator arrow that points in the direction of the most probable next neighbor (as planned by the RL system)
        self.qIndicator=CogArrow()

        # If not otherwise defined or inhibited, each node is also a starting node
        self.startNode=False
        
        # This reward bias is assigned to the node as standard (0.0), and can be changed dynamically to reflect environmental reconfigurations of rewards
        self.nodeRewardBias=0.0
        
