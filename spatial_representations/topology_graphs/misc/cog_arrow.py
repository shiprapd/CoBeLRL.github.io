import numpy as np
import PyQt5 as qt
import pyqtgraph as qg

from PyQt5 import QtGui



class CogArrow(qg.ArrowItem):
    """
    Helper class for the visualization of the topology graph.It constructs a
    centered arrow pointing in a dedicated direction, inherited from 'ArrowItem'
    """

    
    def setData(self,x,y,angle):
        """
        Set the position and direction of the arrow

        :param x: x position of the arrow's center
        :param y: y position of the arrow's center
        :param angle: the orientation of the arrow
        :return: None
        """
        
        # The angle has to be modified to suit the demands of the environment(?)
        # angle = -angle/np.pi*180.0+180.0
        angle = 180.0 - angle
        # Assemble a new temporary dict that is used for path construction
        tempOpts=dict()
        tempOpts['headLen']=self.opts['headLen']
        tempOpts['tipAngle']=self.opts['tipAngle']
        tempOpts['baseAngle']=self.opts['baseAngle']
        tempOpts['tailLen']=self.opts['tailLen']
        tempOpts['tailWidth']=self.opts['tailWidth']
        
        arrowPath=qg.functions.makeArrowPath(**tempOpts)     # Create the path
        bounds=arrowPath.boundingRect()    # Boundaries of arrows, required to shif the arrow
        transform=QtGui.QTransform()
        transform.rotate(angle)
        transform.translate(int(-float(bounds.x())-float(bounds.width())/10.0*7.0),int(float(-bounds.y())-float(bounds.height())/2.0))
        self.path=transform.map(arrowPath)    # 'remap' the path
        self.setPath(self.path)
        self.setPos(x,y)            # Set position of the arrow

