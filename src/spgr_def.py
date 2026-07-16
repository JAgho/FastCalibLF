# %% 
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
from console.utilities.sequences.system_settings import system as default_system

# current calibration files are in : "/home/openimaging/Code/openimaging-mri/src/cortex/utilities"

# %%

#PARAMS

"""
params

-between RF and prephaser delay

-between prephaser and readout gradient delay

- between readout and spoiler delay

-TR1 and TR2 

-gaps between gradient lobes in double gradient echo

nb_dummies

nb_averages

"""

# %%
#Sequence loops

"""
for i nb_dummies:
    (Sequence with ADC off)
    
    for g in grad_channel_index:
        
        for a in nb_averages:
            (Sequence with ADC)


"""


# %%


