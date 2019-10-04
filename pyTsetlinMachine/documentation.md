## class MultiClassTsetlinMachine()
  ### function  __init__()
| args        | dtype           | default  | notes |
| ------------- |:-------------:| :-----:|----------------|
| self |  |  |  |
| number_of_clauses | int | none | | 
| T | int | none | threshold | 
| s | float | none | threshold | 
| boost_true_positive_feedback | int | 1 | possible values=[0,1] | 
| number_of_state_bits | int | 8 |  | 
 
  #### output: tm
  
### function fit()
 | args        | dtype           | default  | notes |
| ------------- |:-------------:| :-----:|----------------|
| self |  |  |  |
| X |  |  |  |
| Y |  |  |  |
| epochs | int | 100 |  |
| incremental | boolean | False |  |

 #### output: tm
 
 ### function predict()
 | args        | dtype           | default  | notes |
| ------------- |:-------------:| :-----:|----------------|
| self |  |  |  |
| X |  |  |  |

#### output: Y
