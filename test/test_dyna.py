
# add current module to search path
import os
import sys
sys.path.append( os.path.join(os.path.realpath(__file__),"..") )


import unittest2 as unittest
from qd.cae.dyna import *


class TestDynaModule(unittest.TestCase):
    """Tests the dyna module."""

    def test_dyna_d3plot(self):
        """Test all D3plot functions"""

        d3plot_filepath = "test/d3plot"
        d3plot_modes = ["in","mid","out","max","min","mean"]
        d3plot_vars = ["disp","vel","accel",
                       "stress","strain","plastic_strain",
                       "history 1 shell","history 1 solid"]

        ## D3plot loading
        d3plot = D3plot(d3plot_filepath)
        for mode in d3plot_modes: # load every var with every mode
            d3plot = D3plot(d3plot_filepath)
            vars2 = ["%s %s" % (var,mode) for var in d3plot_vars]
            for var in vars2:
                d3plot.read_states(var)
            d3plot = D3plot(d3plot_filepath,read_states=vars2) # all at once
        d3plot = D3plot(d3plot_filepath,read_states=d3plot_vars) # default mode (mode=mean)
        # D3plot functions
        self.assertEqual( d3plot.get_nNodes()        ,4915)
        self.assertEqual( len(d3plot.get_nodes())    ,4915)
        self.assertEqual( d3plot.get_nElements()     ,4696)
        self.assertEqual( len(d3plot.get_elements()) ,4696)
        self.assertEqual( d3plot.get_timesteps()[0]  ,0.)
        self.assertEqual( len(d3plot.get_timesteps()) ,1)
        ## D3plot error handling
        # ... TODO

        ## Node
        node_ids = [1,2]
        nodes_ids_v1 = [d3plot.get_nodeByID(node_id) for node_id in node_ids]
        nodes_ids_v2 = d3plot.get_nodeByID(node_ids)
        self.assertItemsEqual( [node.get_id() for node in nodes_ids_v1] , node_ids )
        self.assertItemsEqual( [node.get_id() for node in nodes_ids_v2] , node_ids )
        for node in nodes_ids_v1:
            self.assertEqual( len(node.get_coords())   , 3 )
            self.assertEqual( len(node.get_coords(-1)) , 3 )
            self.assertEqual( len(node.get_disp())     , 1 )
            self.assertEqual( len(node.get_vel())      , 1 ) 
            self.assertEqual( len(node.get_accel())    , 1 ) 
            self.assertGreater( len(node.get_elements()) , 0)

        node_indexes = [0,1]
        node_matching_ids = [1,2] # looked it up manually
        self.assertEqual( len(node_indexes) , len(node_matching_ids) )
        nodes_indexes_v1 = [d3plot.get_nodeByIndex(node_index) for node_index in node_indexes]
        nodes_indexes_v2 = d3plot.get_nodeByIndex(node_indexes)
        self.assertItemsEqual( [node.get_id() for node in nodes_indexes_v1] , node_matching_ids )
        self.assertItemsEqual( [node.get_id() for node in nodes_indexes_v2] , node_matching_ids )
        # .. TODO Error stoff
        
        ## Shell Element
        element_ids = [1,2]
        element_ids_shell_v1 = [d3plot.get_elementByID("shell", element_id) for element_id in element_ids]
        element_ids_shell_v2 = d3plot.get_elementByID("shell", element_ids)
        self.assertItemsEqual( [element.get_id() for element in element_ids_shell_v1] , element_ids )
        self.assertItemsEqual( [element.get_id() for element in element_ids_shell_v2] , element_ids )
        for element in element_ids_shell_v1:
            pass
        # .. TODO Error stoff
        


    def test_binout(self):
        """Testing all Binout functions."""

        binout_filepath = "test/binout"
        content = ["swforc"]
        content_subdirs = [ ['ids','axial', 'failure_time', 'failure', 'length', 'resultant_moment', 'shear'] ]
        nTimesteps = 321
        
        binout = Binout(binout_filepath)
        self.assertEqual( len(binout.get_labels()) , 1)
        self.assertItemsEqual( content , binout.get_labels() )
        
        for content_dir, content_subdirs in zip(content,content_subdirs):
            self.assertItemsEqual( content_subdirs , binout.get_labels(content_dir) )
            for subdir in content_subdirs:
                if subdir == "ids":
                    binout.get_data(content_dir,subdir)
                else:
                    time,data = binout.get_data(content_dir,subdir)
                    self.assertEqual( len(time) , nTimesteps )
                    self.assertEqual( len(time) , len(data) )


if __name__ == "__main__":
    unittest.main()