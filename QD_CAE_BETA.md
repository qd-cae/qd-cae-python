
# Module qd.cae.beta

This module contains functions and classes regarding software from Beta-CAE Systems.

classes: 
  - [MetaCommunicator](#MetaCommunicator)

functions: None

submodules: None

---------
# Classes

The modules classes.

## MetaCommunicator

A class in order to communicate with an META from Beta-CAE Systems.

### ```MetaCommunicator(meta_path=None, ip_address="127.0.0.1", meta_listen_port=4342)```

Construct a MetaCommunicator
The constructor checks for a running and listening instance of META and 
connects to it. If there is no running instance it will start one and will 
wait for it to be ready to operate. One has two options to let python know
where the META-executable is: give a path as argument or set the environment
variable META_PATH.

Parameters:
  - str meta_path : optional path to Meta
  - str ip_address : ip-adress to connect to. Localhost by default.
  - int meta_listen_port : port on which meta is listening (or in case of startup will listen)

Example:
```python
from qd.cae.beta import MetaCommunicator
mc = MetaCommunicator()
```

### ```is_runnning(timeout_seconds=None)```

Check whether META is up running.

Parameters:
  - int timeout_seconds : seconds of waiting until timeout

Returns:
  - bool is_running : True if META is running. False otherwise.

Example:
```python
from qd.cae.beta import MetaCommunicator
mc = MetaCommunicator()
mc.is_running()
# >>> True
```


### ```send_command(command, timeout=20)```

Send a command to META.

Parameters:
  - str command : command to send to META
  - int timeout : 

Example:
```python
from qd.cae.beta import MetaCommunicator
mc = MetaCommunicator()
mc.send_command("read geom auto path/to/d3plot")
```

### ```show_pids(partlist=None, show_only=False)```

Tell META to make parts visible. 
Shows all pids by default. If partlist is given, META performs
a show command for these pids. If show_only is used, all other
parts will be removed from vision.

Parameters:
  - list(int) partlist : list of part ids
  - bool show_only : whether to show only these parts (removes all others)

Example:
```python
from qd.cae.beta import MetaCommunicator
mc = MetaCommunicator()
mc.show_pids( [1,2,3], show_only=True)
```

### ```hide_pids(partlist=None)```

Tell META to make parts invisible. 
Hides all pids by default. If partlist is given, META performs
a show command for these pids. 

Parameters:
  - list(int) partlist : list of part ids

Example:
```python
from qd.cae.beta import MetaCommunicator
mc = MetaCommunicator()
mc.hide_pids( [1,2,3] )
```

### ```read_geometry(filepath)```

Read the geometry from a file into META.

Parameters:
  - str filepath : path to the result file

Example:
```python
from qd.cae.beta import MetaCommunicator
mc = MetaCommunicator()
mc.read_geometry("path/to/d3plot")
```

### ```read_d3plot(filepath)```

Open a d3plot in META and read geometry, displacement and effective plastic-strain at once.

Parameters:
  - str filepath : path to d3plot result file

Example:
```python
from qd.cae.beta import MetaCommunicator
mc = MetaCommunicator()
mc.read_d3plot("path/to/d3plot")
```
