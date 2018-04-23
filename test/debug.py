from qd.cae.dyna import *

d3plot = D3plot ("test/d3plot", read_states = ["vel","accel"])

print (d3plot.get_node_velocity().shape)
print (d3plot.get_node_acceleration().shape)
print (d3plot.get_node_coords().shape)
print (d3plot.get_node_ids().shape)
help(d3plot.get_node_velocity)
help(d3plot.get_node_acceleration)
help(d3plot.get_node_ids)
