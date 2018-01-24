
import os
import time
import numbers
import socket
import platform
import subprocess

class MetaCommunicator:
    '''A class in order to communicate with META from Beta-CAE Systems
    '''

    def __init__(self, meta_path=None, ip_address="127.0.0.1", meta_listen_port=4342):
        '''Constructor for a MetaCommunicator

        Parameters
        ----------
        meta_filepath : str
            optional path or command for the META executable
        ip_address : str
            ip-adress to connect to. Localhost by default.
        meta_listen_port : int
            port on which meta is listening (or in case of startup will listen)

        Notes
        -----
            The constructor checks for a running and listening instance of META and 
            connects to it. I   there is no running instance it will start one and will 
            wait for it to be ready to operate.
            If META has to be started, either provide a filepath or simply set the 
            environment variable META_PATH to the executable.

        Examples
        --------
            >>> mc = MetaCommunicator() # starts META , if not listening
            >>> mc.is_runnning()
            True
            >>> # Send arbitrary Meta commands ... hihihi
            >>> mc.send_command("read geom auto path/to/file")
            >>> mc.show_pids( [11,12,13] ) # show only 3 pids
            >>> mc.hide_pids() # hide all
            >>> mc.show_pids() # show all
        '''

        assert meta_listen_port >= 0

        self.ip_address = ip_address
        self.meta_listen_port = meta_listen_port
        self.address = "%s@%s" % (str(self.meta_listen_port),self.ip_address)

        # run new META of not listening
        if not self.is_running():
            if meta_path:
                self.meta_path = meta_path
            else:
                try:
                    self.meta_path = os.environ["META_PATH"]
                except KeyError as e:
                    raise RuntimeError("Please either give the communicator a path to META or set the environment variable META_PATH to the executable.")

        # remote path executable for communication
        # this was stolen from the distribution 16.1.2
        self.meta_remote_control_path = os.path.dirname(os.path.realpath(__file__))
        if platform.system() == "Windows":
            self.meta_remote_control_path = os.path.join(self.meta_remote_control_path,'meta_remote_control.exe')
        else:
            self.meta_remote_control_path = os.path.join(self.meta_remote_control_path,'meta_remote_control')
			
        # check for running instance
        if not self.is_running():
            self._start() # blocks until meta started!


    def _start(self):
        '''Start META with predefind settings
        '''

        cmd = "%s %s %s %s" % (self.meta_path,'-nolauncher',"-listenport",str(self.meta_listen_port))
        subprocess.Popen(cmd)

        if not self.is_running(timeout_seconds=60):
            raise RuntimeError("Timeout: META did not start within %d seconds." % timeout_seconds)


    def is_running(self,timeout_seconds=None):
        '''Check whether META is up running

        Parameters
        ----------
        timeout_seconds : int
            seconds of waiting until timeout
        
        Returns
        -------
        is_running : bool
            True if META is running. False otherwise

        Examples
        --------
            >>> mc = MetaCommunicator()
            >>> mc.is_running()
            True
            >>> # Pretty unfunny here, but I actually closed META now
            >>> mc.is_running()
            False
        '''

        # no timeout, just check
        if timeout_seconds==None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.ip_address,self.meta_listen_port))
                s.close()
                return True
            except:
                return False
        # in case of a timeout perform multiple tries
        else:
            timeout_seconds = int(timeout_seconds)
            for ii in range(timeout_seconds):
                if self.is_running():
                    return True
                time.sleep(1) # second
            return False

        
    def send_command(self, command, timeout=20):
        '''Send a command to a remote META

        Parameters
        ----------
        command : str
            command to send to META
        timeout : int
            timeout to wait until the command finishes
        
        Notes
        -----
            The commands send over to META are identical as if they would be
            typed into the command line window.

        Examples
        --------
            >>> mc = MetaCommunicator()
            >>> mc.send_command("read geom auto path/to/file")
            >>> mc.send_command("add pid 4,5,3")
        '''

        # check
        if not self.is_running():
            raise RuntimeError("META is not running.")
        
        # send command
        subprocess.call([self.meta_remote_control_path,
                         "-wait",
                         str(timeout),
                         self.address,command])


    def show_pids(self, partlist=None, show_only=False):
        '''Tell META to make parts visible

        Parameters
        ----------
        partlist : list(int)
            list of pids
        show_only : bool
            whether to show only these parts (removes all others)

        Notes
        -----
            Shows all pids by default. If partlist is given, META performs
            a show command for these pids. If show_only is used, all other
            parts will be removed from vision.

        Examples
        --------
            >>> # show all pids
            >>> mc.show_pids()
            >>> # make two pids visible
            >>> mc.show_pids( [13,111] ) 
            >>> # let's show only two pids
            >>> mc.show_pids( [13,111], show_only=True)
        '''

        # filter parts
        partlist = [] if partlist==None else partlist
        if isinstance(partlist, int):
            partlist = [partlist]
        partlist = [ int(entry) for entry in partlist if isinstance(entry, numbers.Number) ]
        
        # clean if neccesary
        if show_only:
            self.hide_pids()
        
        # send command
        if partlist:
            self.send_command("add pid %s" % str(partlist)[1:-1]  )
        else:
            self.send_command("add pid all" )


    def hide_pids(self, partlist=None):
        '''Tell META to make parts invisible. 

        Parameters
        ----------
        partlist : list(int)
            list of pids

        Notes
        -----
            Hides all pids by default. If partlist is given, META performs
            a hide command for these pids. 

        Examples
        --------
            >>> # hide two pids
            >>> mc.hide_pids( [13,111] ) 
            >>> # hide all pids
            >>> mc.hide_pids()
        '''

        # filter parts
        partlist = [] if partlist==None else partlist
        if isinstance(partlist, int):
            partlist = [partlist]
        partlist = [ int(entry) for entry in partlist if isinstance(entry, numbers.Number) ]
        
        # send command
        if partlist:
            self.send_command("era pid %s" % str(partlist)[1:-1]  )
        else:
            self.send_command("era pid all" )


    def read_geometry(self, filepath):
        '''Read the geometry from a file

        Parameters
        ----------
        filepath : str
            path to the result file

        Examples
        --------
            >>> mc.read_geometry("path/to/result/file")
            >>> # yay we see in meta some geometry ... but not here
        '''

        # remove model focus, otherwise active will be deleted
        self.send_command("model active empty")
        self.send_command("read geom auto %s" % filepath)


    def read_d3plot(self,filepath):
        '''Open a d3plot in META and read geometry, displacement and plastic-strain.

        Parameters
        ----------
        filepath : str
            path to d3plot result file

        Examples
        --------
            >>> mc.read_d3plot("path/to/d3plot")
        '''

        # remove model focus, otherwise active will be deleted
        self.send_command("model active empty")
        self.send_command("read geom Dyna3D %s" % filepath)
        self.send_command("read dis Dyna3D %s all Displacements" % filepath)
        self.send_command("read onlyfun Dyna3d %s all Stresses,PlasticStrain,MaxofInOutMid" % filepath)


# test
if __name__ == "__main__":
    pass
