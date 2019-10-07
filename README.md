# annotated_pytsetlin_machine
fork repo

clone/download onto local system

~~~~bash
cd annotated_pytsetlin_machine/pyTsetlinMachine
make
~~~~

## To note as different from CAIR version:
####  in ../pyTsetlinMachine/tm.py 
~~~~ 
_lib = np.ctypeslib.load_library('libTM', os.path.join(this_dir, "."))
~~~~ 

instead of 
~~~~ 
_lib = np.ctypeslib.load_library('libTM', os.path.join(this_dir, ".."))
~~~~ 
_note the single dot in os.path.join

~~~~bash
cd ../examples
~~~~

#### in ../examples/*.py
~~~~ 
import sys
sys.path.append('../pyTsetlinMachine/')
from tm import MultiClassTsetlinMachine
~~~~ 

instead of 
~~~~ 
from pyTsetlinMachine.tm import MultiClassTsetlinMachine 
~~~~ 
