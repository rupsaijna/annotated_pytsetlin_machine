# annotated_pytsetlin_machine
fork repo

clone/download onto local system

### My version :
python 2.7.12  
numpy 1.16.0  
tensorflow 1.14.0  
keras 2.3.0

_HAVE NOT CHECKED FOR OTHER VERSIONS, Please do_

### making the library for local use

~~~~bash
cd annotated_pytsetlin_machine/pyTsetlinMachine
make
~~~~
### Running the NoisyXOR example
~~~~bash
cd ../examples
python NoisyXORDemo.py
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
