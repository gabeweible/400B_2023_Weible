#!/usr/bin/env python
# coding: utf-8

# # Getting the Mass of a Galaxy

# **Imports:**

# In[1]:


from ReadFile import Read # We made this for HW 2
import numpy as np
import astropy.units as u # import astropy units
from astropy import constants as const # import astropy constants
from tabulate import tabulate # Will help us make a table


# ### ComponentMass Function

# In[2]:


def ComponentMass(file, p_type):
    """
    Purpose: return the total mass of any desired galaxy component.
    
    Inputs:
        file : 'string'
            .txt galaxy simulation data file to read in
        p_type : 'int' or 'float'
            particle type, 1 = Halo, 2 = Disk, 3 = Bulge
    
    Returns:
        M : 'astropy quantity'
            Total mass of particle type p_type for the galaxy in file in MSun
            Units of 10^12 MSun, rounded to three decimal places.
    """
    time, total, data = Read(file) # Read in the file
    
    # Restrict our data to only particles of the correct type
    data = data[data['type'] == p_type]
    
    # Sum all the masses of type p_type, and add units of 10^12 MSun
    M = np.sum(data['m'])*(1e10 * u.Msun / 1e12 * u.Unit(1e12))
    
    # round to three decimal points
    M = np.round(M, 3)

    return M # Return the total mass for the given component


# **Let's use this now:**

# In[3]:


# # Milky Way component masses
# MW_halo = ComponentMass('MW_000.txt', 1)
# MW_disk = ComponentMass('MW_000.txt', 2)
# MW_bulge = ComponentMass('MW_000.txt', 3)

# # M31 (Andromeda) component masses
# M31_halo = ComponentMass('M31_000.txt', 1)
# M31_disk = ComponentMass('M31_000.txt', 2)
# M31_bulge = ComponentMass('M31_000.txt', 3)

# # M33 (Triangulum) component masses, no bulge
# M33_halo = ComponentMass('M33_000.txt', 1)
# M33_disk = ComponentMass('M33_000.txt', 2)
# M33_bulge = 0 * u.Unit(1e12) * u.Msun # Set to 0 and keep units for math purposes

# # Print out our MW values to see if they make sense 
# # (3 decimal pts. incl. final zeroes)
# print(f"Milky Way halo: {MW_halo:.3f}")
# print(f"Milky Way disk: {MW_disk:.3f}")
# print(f"Milky Way bulge: {MW_bulge:.3f}\n")

# # Same for M31 / Andromeda (3 decimal pts. incl. final zeroes)
# print(f"M31 halo: {MW_halo:.3f}")
# print(f"M31 disk: {M31_disk:.3f}")
# print(f"M31 bulge: {M31_bulge:.3f}\n")

# # And for M33 / Triangulum (3 decimal pts. incl. final zeroes)
# print(f"M33 Way halo: {M33_halo:.3f}")
# print(f"M33 disk: {M33_disk:.3f}")
# print(f"M31 bulge: {M33_bulge:.3f}")


# **More calculations:**

# In[5]:


# # Total masses (just add all three components for each):
# MW_tot = MW_halo + MW_disk + MW_bulge
# M31_tot = M31_halo + M31_disk + M31_bulge
# M33_tot = M33_halo + M33_disk + M33_bulge

# # For the local group (all three together, sum by component)
# LG_halo = MW_halo + M31_halo + M33_halo
# LG_disk = MW_disk + M31_disk + M33_disk
# LG_bulge = MW_bulge + M31_bulge + M33_bulge
# LG_tot = LG_halo + LG_disk + LG_bulge # Add 'em all up

# # Baryon fractions (total stellar mass / total mass)
# MW_f = (MW_disk + MW_bulge) / MW_tot
# M31_f = (M31_disk + M31_bulge) / M31_tot
# M33_f = (M33_disk + M33_bulge) / M33_tot
# LG_f = (LG_disk + LG_bulge) / LG_tot


# ### Now let's make a table:

# In[7]:


# # List of table header strings (w/units typeset in LaTeX, where applicable)
# head = ['Galaxy Name', r'Halo Mass ($10^{12} \, \mathrm{M}_{\odot}$)',
#         r'Disk Mass ($10^{12} \, \mathrm{M}_{\odot}$)', r'Bulge Mass ($10^{12} \, \mathrm{M}_{\odot}$)',
#         r'Total ($10^{12} \, \mathrm{M}_{\odot}$)', r'$f_{\mathrm{bar}}$']

# # Milky Way row, and make sure we keep trailing zeroes on rounded values
# MW = ['Milky Way', *(f'{i.value:.3f}' for i in [MW_halo, MW_disk, MW_bulge, MW_tot, MW_f])]

# # Same for Andromeda / M31
# M31 = ['M31 (Andromeda)', *(f'{i.value:.3f}' for i in [M31_halo, M31_disk, M31_bulge, M31_tot, M31_f])]

# # Same for Triangulum / M33
# M33 = ['M33 (Triangulum)', *(f'{i.value:.3f}' for i in [M33_halo, M33_disk, M33_bulge, M33_tot, M33_f])]

# # And finally, for all three together (the Local Group)
# LG = ['Local Group (All 3)', *(f'{i.value:.3f}' for i in [LG_halo, LG_disk, LG_bulge, LG_tot, LG_f])]

# # Put all the rows into a table (each element of this list is itself a list)
# table =  [head, MW, M31, M33, LG]

# # Print LaTeX for the table (will need to modify slightly, probably.)
# # kwargs: first row is the header, we want straight-up LaTeX, and keep our numbers how they are
# print(tabulate(table, headers='firstrow', tablefmt='latex_raw', disable_numparse=True))


# **The above printed LaTeX will yield a nice table, albeit needing some minor cosmetic modfications to preference.**
