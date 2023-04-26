#!/usr/bin/env python
# coding: utf-8

# # Let's make a program that will read in our .txt Milky Way data files!

# Imports:

# In[3]:


import numpy as np
import astropy.units as u


# Define a *Read* function:

# In[4]:


def Read(filename):
    """
    Purpose: open and read a Milky Way simulation .txt data file
    
    Input: .txt data filename to read in
    
    Returns: simulation time, total number of particles, and a data
    array containing particle type, mass, and 3D positions and
    velocities (in that order)
    """
    file = open(filename, 'r') # 'r' is for read :)
    
    # First line gives us the time of the snapshot
    line1 = file.readline()
    label, value = line1.split() # We only really need the value
    time = float(value) * u.Myr # Slap on units of Myr
    
    # Second line gives us the _total_ number of particles
    line2 = file.readline()
    label, value = line2.split() # Again, only really need the value
    total = float(value) # Dimensionless => unitless
    
    file.close() # Close the file (duh)
    
    # Get all of our particle data, skipping the 3 lines of header
    # and including labels (dtypes assigned automatically)
    data = np.genfromtxt(filename, dtype=None, names=True, skip_header=3)
    
    return time, total, data # We're done *fire emoji*


# ## Testing:

# ***Note: this will need to be commented out before creating a .py script from the notebook for the purpose of importing this function***

# In[11]:


# # Read in our t = 0 Myr data file
# test_t, test_tot, test_dat = Read('MW_000.txt')

# print(test_t, test_tot)
# print(test_dat['type'][1]) # 2nd particle type
# print(test_dat['m'][0]) # First particle mass
# print(test_dat['x'][2]) # Third particle x-coord


# All right, I think that seems to work well enough, the test will be commented out now.
