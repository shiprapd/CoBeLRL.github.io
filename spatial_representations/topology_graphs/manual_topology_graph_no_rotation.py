import PyQt5 as qt
import pyqtgraph as qg
import pyqtgraph.functions
import numpy as np
import gym
from PyQt5 import QtGui

from .misc.topology_node import TopologyNode
from .misc.cog_arrow import CogArrow
from spatial_representations.spatial_representation import SpatialRepresentation



class ManualTopologyGraphNoRotation(SpatialRepresentation):
    #TODO : should not take modules as input, specific inputs
    
    def __init__(self, modules, graph_info):
        """
        This initializes and sets up the manually constructed topology graph with no rotation.

        :param modules: A dictionary containing all employed modules
        :param graph_info: Graph information containing start node, goal node and clique size
        :return: None
        """
        
        super(ManualTopologyGraphNoRotation,self).__init__()    # Call the base class init
        
        # Normally, the topology graph is not shown in gui_parent
        self.visual_output=False
        self.gui_parent=None
        self.graph_info=graph_info    # Store the graph parameters
        self.modules=modules    # Extract the world module
    
        #TODO : if world_module is not None : 
        world_module=modules['world']
        world_module.setTopology(self)
        
        self.world_limits = world_module.getLimits()    # Get the limits of the given environment 
        
        # Retrieve all boundary information from the environment
        self.world_nodes, self.world_edges = world_module.getWallGraph() 
        
        # Inherent definitions for the topology graph
        # This is the node corresponding to the robot's actual position
        self.currentNode=-1
        # This is the node corresponding to the robot's next position
        self.nextNode=-1
        # This list of topologyNode[s] stores all nodes of the graph
        self.nodes=[]
        # This list of [int,int]-entries defines all edges (unique) which make up the graph's connectivity
        self.edges=[]
        self.cliqueSize=graph_info['cliqueSize']

        # Set up a manually constructed topology graph
        # Read topology structure from world module
        nodes=np.array(world_module.getManuallyDefinedTopologyNodes())
        nodes=nodes[nodes[:,0].argsort()]
        edges=np.array(world_module.getManuallyDefinedTopologyEdges())
        edges=edges[edges[:,0].argsort()]
        
        indexCounter=0
        for n in nodes:
            # Create the corresponding node, where i is the running index of the mesh_points/corresponding nodes
            node=TopologyNode(indexCounter,float(n[1]),float(n[2]))
            self.nodes=self.nodes+[node]
            indexCounter+=1
        
        # Fill in the self.edges list from the edges information
        for e in edges:
            self.edges=self.edges+[[int(e[1]),int(e[2])]]
                
        # Define a dedicated 'noneNode' that acts as a placeholder for neighborhood construction
        noneNode=TopologyNode(-1,0.0,0.0)
        
        # Construct the neighborhoods of each node in the graph
        for edge in self.edges:
            # First edge node
            a=self.nodes[int(edge[0])]
            # Second edge node
            b=self.nodes[int(edge[1])]
            # Add node a to the neighborhood of node b, and vice versa
            a.neighbors=a.neighbors+[b]
            b.neighbors=b.neighbors+[a]
        
        # It is possible that a node does not have the maximum possible number of neighbors, 
        # to stay consistent in RL, fill up the neighborhood with noneNode[s]:
        for node in self.nodes:
            while len(node.neighbors)<self.cliqueSize:
                node.neighbors=node.neighbors+[noneNode]
        
        for nodeIndex in self.graph_info['startNodes']:
            self.nodes[nodeIndex].startNode=True
        
        for nodeIndex in self.graph_info['goalNodes']:
            self.nodes[nodeIndex].goalNode=True
        
        #TODO : Test : Remove from class definition if it is only being used for visualization
        self.sample_state_space()
        



    def set_visual_debugging(self,visual_output,gui_parent):
        """
        This method set up the Visual elements for debugging in the output window.

        :param visual_output: Boolean value (True or False)
        :param gui_parent: Output window
        :return: None
        """

        self.gui_parent=gui_parent
        self.visual_output=visual_output
        self.initVisualElements()
        



    def initVisualElements(self):
        """
        This method initializes the Visual elements for debugging in the output window.

        :param: None
        :return: None
        """

        if self.visual_output:
            self.plot = self.gui_parent.addPlot(title='Topology graph')
            # Set extension of the plot
            self.plot.setXRange( self.world_limits[0,0], self.world_limits[0,1] )
            self.plot.setYRange( self.world_limits[1,0], self.world_limits[1,1] )
            self.plot.setAspectLocked()    # Lock aspect ratio

            # Set up indicator arrows for each node, except the goal node, and all nodes in active shock zones if shock zones exist
            for node in self.nodes:
                if not node.goalNode:
                    node.qIndicator=CogArrow(angle=0.0,headLen=20.0,tipAngle=25.0,tailLen=0.0,brush=(255,255,0))
                    self.plot.addItem(node.qIndicator)
                
            self.perimeterGraph=qg.GraphItem()    # Overlay the world's perimeter
            self.plot.addItem(self.perimeterGraph)
            
            self.perimeterGraph.setData(pos=np.array(self.world_nodes),adj=np.array(self.world_edges),brush=(128,128,128))
            
            self.topologyGraph=qg.GraphItem()    # Overlay the topology graph
            self.plot.addItem(self.topologyGraph)
            
            # Set up a brushes array for visualization of the nodes. Normal nodes are grey
            symbolBrushes=[qg.mkBrush(color=(128,128,128))]*len(self.nodes)

            for node in self.nodes:
                if node.startNode:
                    symbolBrushes[node.index]=qg.mkBrush(color=(0,255,0))    # Green

                if node.goalNode:
                    symbolBrushes[node.index]=qg.mkBrush(color=(255,0,0))    # Red
        
            tempNodes=[]
            tempEdges=[]
        
            for node in self.nodes:
                tempNodes=tempNodes+[[node.x,node.y]]
                
            for edge in self.edges:
                tempEdges=tempEdges+[[edge[0],edge[1]]]

            self.topologyGraph.setData(pos=np.array(tempNodes),adj=np.array(tempEdges),symbolBrush=symbolBrushes)
            self.posMarker=CogArrow(angle=0.0,headLen=20.0,tipAngle=25.0,tailLen=0.0,brush=(255,0,0))    # Overlay robot marker
            self.plot.addItem(self.posMarker)                # Initial position to center
            self.posMarker.setData(0.0,0.0,0.0)
            
        

    def updateVisualElements(self):
        """
        This method updates the Visual elements for debugging in the output window.

        :param: None
        :return: None
        """
        #TODO : make different parts of visualization optional overlay the policy arrows
        
        if self.visual_output:
            #TODO : sample state space here
            
            for node in self.nodes:
                # Query the model at each node's position, only for valid nodes!
                if node.index!=-1:
                    observation=self.state_space[node.index]
                    data=np.array([[observation]])
                    # Get the q-values at the queried node's position
                    q_values = self.rlAgent.agent.model.predict_on_batch(data)[0]
                    
                    # Find all neighbors that are actually valid (index != -1)
                    validIndex=0
                    for n_index in range(len(node.neighbors)):
                        if node.neighbors[n_index].index!=-1:
                            validIndex=n_index
                    
                    # Find the index of the neighboring node that is 'pointed to' by the highest q-value, AND is valid!
                    maxNeighNode=node.neighbors[np.argmax(q_values[0:validIndex+1])]
                    # Find the direction of the selected neighboring node
                    # To node: maxNeighNode
                    toNode=np.array([maxNeighNode.x,maxNeighNode.y])
                    # From node: node
                    fromNode=np.array([node.x,node.y])
                    # The difference vector between to and from
                    vec=toNode-fromNode
                    # Normalize the direction vector
                    l=np.linalg.norm(vec)
                    vec=vec/l
                    # Make the corresponding indicator point in the direction of the difference vector
                    node.qIndicator.setData(node.x,node.y,np.rad2deg(np.arctan2(vec[1],vec[0])))  



    def updateRobotPose(self,pose):
        """
        This function updates the visual depiction of the agent(robot)

        :param pose: The agent's pose to visualize
        :return: None
        """

        if self.visual_output:
            self.posMarker.setData(pose[0],pose[1],np.rad2deg(np.arctan2(pose[3],pose[2])))




    def sample_state_space(self):
        """
        This method generates a sample state space
        In this topology graph, a state is an image sampled from a specific node of the graph.
        There is no rotation, so one image per node is sufficient.

        :param: None
        :return: None
        """
        
        # TODO : test what this does
        world_module=self.modules['world']
        observation_module=self.modules['observation']
        
        
        self.state_space=[]
        for node_index in range(len(self.nodes)):
            node=self.nodes[node_index]
            # set agent to x/y-position of 'node'
            world_module.step_simulation_without_physics(node.x,node.y,90.0)
            world_module.step_simulation_without_physics(node.x,node.y,90.0)
            observation_module.update()
            observation=observation_module.observation
            self.state_space+=[observation]
        return




    def generate_behavior_from_action(self,action):
        """
        According to the action provided (reset or other), generate the behavior.
        Not reset: If next node is invalid node the agent stays in place else moves to the
        next node.
        If reset: The agent is placed at a random node.

        :param action: Contains the action to be performed
        :return callback_value: A dictionary containing the current node
        """
        
        nextNodePos=np.array([0.0,0.0])
        callback_value=dict()
        
        if action!='reset':    # If a standard action is performed
            previousNode=self.currentNode
            # Compute next node with the given action
            # TODO :remove dependence on same module
            self.nextNode=self.nodes[self.currentNode].neighbors[action].index
                
            if self.nextNode!=-1:
                # Compute the next node's coordinates
                nextNodePos=np.array([self.nodes[self.nextNode].x,
                                      self.nodes[self.nextNode].y])
            else:
                # If the next node corresponds to an invalid node, the agent stays in place
                self.nextNode=self.currentNode
                # Prevent the agent from starting any motion pattern
                self.modules['world'].goalReached=True
                nextNodePos=np.array([self.nodes[self.currentNode].x,    
                                      self.nodes[self.currentNode].y])    # Setting next node
            
            # TODO : make callbacks not mandatory
            callback_value['currentNode']=self.nodes[self.nextNode]    # Setting current node to next node
            
        else:
            # A random node is chosen to place the agent at (this node MUST NOT be the global goal node!)
            nextNode=-1
            while True:
                nrNodes=len(self.nodes)
                nextNode=np.random.random_integers(0,nrNodes-1)
                if self.nodes[nextNode].startNode:
                    break
            
            nextNodePos=np.array([self.nodes[nextNode].x,self.nodes[nextNode].y])
            self.nextNode=nextNode
            
        # Move robot to the node
        self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],90.0])) 
        self.modules['world'].actuateRobot(np.array([nextNodePos[0],nextNodePos[1],90.0])) 
        
        self.currentNode=self.nextNode
        self.modules['observation'].update()
        self.updateRobotPose([nextNodePos[0],nextNodePos[1],0.0,1.0])

        # If possible try to update the visual debugging display
        if hasattr(qt.QtGui, 'QApplication'):
            if qt.QtGui.QApplication.instance() is not None:
                qt.QtGui.QApplication.instance().processEvents()
        else:
            if qt.QtWidgets.QApplication.instance() is not None:
                qt.QtWidgets.QApplication.instance().processEvents()
        
        return callback_value



    def get_action_space(self):
        """
        Get action space according to the clique size of the topology graph

        :param: None
        :return: Defined space containing discrete points according to the clique size
        """
        return gym.spaces.Discrete(self.cliqueSize)





