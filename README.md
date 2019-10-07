# annotated_pytsetlin_machine
fork repo
clone onto system

cd pyTsetlinMachine
make

check if code changes are propoer (tm.py -- np.load has to be single .)

cd ../examples

~~~~ 
import sys
sys.path.append('../pyTsetlinMachine/')
from tm import MultiClassTsetlinMachine
~~~~ 

instead of 
~~~~ from pyTsetlinMachine.tm import MultiClassTsetlinMachine ~~~~ 
