#!/usr/bin/env python
# coding: utf-8

# # Homework 4
# ### Center of Mass Position and Velocity
# ### Gabe Weible

# In[44]:


# import modules
import numpy as np # math
import astropy.units as u # units
from ReadFile import Read # we made this!


# In[45]:


class CenterOfMass:
    """
    Purpose: class to define center of mass (CoM) position and velocity properties of a
    given galaxy and simulation snapshot. Get snapshot time, total # of particles, and
    data when initialized, as well as storing indices for the given particle type and
    gives their masses, positions, and velocities. Has 3 methods, one to generally get a
    center of mass for a collection of vectors, one to get the galaxy's CoM position for
    the given ptype, and the velocity of the CoM for that ptype.

    Inputs:
        filename : 'str'
            snapshot .txt filename to get the CoM for
        ptype : 'int'
            1 = Halo, 2 = Disk, 3 = Bulge; particle type to get CoM for
    """

    def __init__(self, filename, ptype):
        ''' Class to calculate the 6-D phase-space position of a galaxy's center of mass using
        a specified particle type. 
            
            PARAMETERS
            ----------
            filename : `str`
                snapshot file
            ptype : `int; 1, 2, or 3`
                particle type to use for COM calculations
        '''
     
        # read data in the given file using Read
        self.time, self.total, self.data = Read(filename)                                                                                             

        #create an array to store indices of particles of desired Ptype                                
        self.index = np.where(self.data['type'] == ptype)

        # store the mass, positions, velocities of only the particles of the given type
        self.m = self.data['m'][self.index]
        self.x = self.data['x'][self.index]
        self.y = self.data['y'][self.index]
        self.z = self.data['z'][self.index]
        self.vx = self.data['vx'][self.index]
        self.vy = self.data['vy'][self.index]
        self.vz = self.data['vz'][self.index]


    def COMdefine(self,a,b,c,m):
        ''' Method to compute the COM of a generic vector quantity by direct weighted averaging.
        
        PARAMETERS
        ----------
        a : `float or np.ndarray of floats`
            first vector component
        b : `float or np.ndarray of floats`
            second vector component
        c : `float or np.ndarray of floats`
            third vector component
        m : `float or np.ndarray of floats`
            particle masses
        
        RETURNS
        -------
        a_com : `float`
            first component on the COM vector
        b_com : `float`
            second component on the COM vector
        c_com : `float`
            third component on the COM vector
        '''
        # xcomponent Center of mass
        a_com = np.sum(a*m) / np.sum(m)
        # ycomponent Center of mass
        b_com = np.sum(b*m) / np.sum(m)
        # zcomponent Center of mass
        c_com = np.sum(c*m) / np.sum(m)
        
        # return the 3 components separately
        return a_com, b_com, c_com
    
    
    def COM_P(self, delta):
        '''Method to compute the position of the center of mass of the galaxy 
        using the shrinking-sphere method.

        PARAMETERS
        ----------
        delta : `float, optional`
            error tolerance in kpc. ``Default'' is 0.1 kpc
        
        RETURNS
        ----------
        p_COM : `np.ndarray of astropy.Quantity'
            3-D position of the center of mass in kpc
        '''                                                                     
        # Center of Mass Position                                                                                      
        ###########################                                                                                    

        # Try a first guess at the COM position by calling COMdefine                                                   
        x_COM, y_COM, z_COM = self.COMdefine(self.x, self.y, self.z, self.m)
        # compute the magnitude of the COM position vector.
        r_COM = np.sqrt(x_COM**2 + y_COM**2 + z_COM**2)


        # iterative process to determine the center of mass                                                      

        # change reference frame to the (first-guess) COM frame                                                                                                                              
        x_new = self.x - x_COM
        y_new = self.y - y_COM
        z_new = self.z - z_COM
        r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

        # find the max 3D distance of all particles from the guessed COM                                               
        # will re-start at half that radius (reduced radius)                                                           
        r_max = max(r_new) / 2.0
        
        # initial value for the change in COM position                                                      
        # between the first guess above and the new one computed from half that volume
        # (will be updated with real values needs to be greater than delta originally)
        change = 1000.0 # kpc

        # start iterative process to determine center of mass position                                                 
        # delta is the tolerance for the difference between the old COM and the new one.    
        
        while (change > delta):
            
            # select all particles within the reduced radius (starting from original x,y,z, m)
            index2 = np.where(r_new < r_max)
            x2 = self.x[index2]
            y2 = self.y[index2]
            z2 = self.z[index2]
            m2 = self.m[index2]

            # Refined COM position:                                                                                    
            # compute the center of mass position using the particles in the half radius
            x_COM2, y_COM2, z_COM2 = self.COMdefine(x2, y2, z2, m2)
            # compute the new 3D COM position
            r_COM2 = np.sqrt(x_COM2**2 + y_COM2**2 + z_COM2**2)

            # determine the difference between the two COM positions                                                                                  
            change = np.abs(r_COM - r_COM2)                                                                  

            # Before loop continues, reset : r_max, particle separations and COM                                                                                                                        
            
            # Change the frame of reference to the new COM with the half radius                                               
            x_new = self.x - x_COM2
            y_new = self.y - y_COM2
            z_new = self.z - z_COM2
            r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

            # set the center of mass positions to the refined values                                                   
            x_COM = x_COM2
            y_COM = y_COM2
            z_COM = z_COM2
            r_COM = r_COM2

            # create an array (np.array) to store the COM position                                                                                                                                                       
            p_COM = np.array([x_COM, y_COM, z_COM])
            
            # reduce the volume by a factor of 2 again (for the next loop iteration)                                                                
            r_max /= 2.0

        # set the correct units using astropy and round all values
        # and then return the COM positon vector
        return np.around(p_COM, decimals=2) * u.kpc
        
        
        
    def COM_V(self, x_COM, y_COM, z_COM):
        ''' Method to compute the center of mass velocity based on the center of mass
        position.

        PARAMETERS
        ----------
        x_COM : 'astropy quantity'
            The x component of the center of mass in kpc
        y_COM : 'astropy quantity'
            The y component of the center of mass in kpc
        z_COM : 'astropy quantity'
            The z component of the center of mass in kpc
            
        RETURNS
        -------
        v_COM : `np.ndarray of astropy.Quantity'
            3-D velocity of the center of mass in km/s
        '''
        
        # the max distance from the center that we will use to determine 
        #the center of mass velocity                   
        rv_max = 15.0 # kpc

        # determine the position of all particles relative to the center of mass position (x_COM, y_COM, z_COM)
        # remove units for now, then add at the end again.
        xV = self.x - (x_COM / u.kpc)
        yV = self.y - (y_COM / u.kpc)
        zV = self.z - (z_COM / u.kpc)
        rV = np.sqrt(xV**2 + yV**2 + zV**2)
        
        # determine the indices for those particles within the max radius (rel. to CoM)
        indexV = np.where(rV < rv_max)
        
        # determine the velocities and masses of those particles within the max radius
        vx_new = self.vx[indexV]
        vy_new = self.vy[indexV]
        vz_new = self.vz[indexV]
        m_new = self.m[indexV]
        
        # compute the center of mass velocity using those particles
        vx_COM, vy_COM, vz_COM = self.COMdefine(vx_new, vy_new, vz_new, m_new)
        
        # create an np.array to store the COM velocity
        v_COM = np.array([vx_COM, vy_COM, vz_COM])

        # return the COM vector
        # set the correct units using astropy
        # round all values
        return np.around(v_COM, decimals=2) * u.km/u.s


# In[46]:


# # Create a Center of mass object for the MW, M31 and M33
# # below is an example of using the class for MW
# MW_COM = CenterOfMass("MW_000.txt", 2)


# In[47]:


# # below gives you an example of calling the class's functions
# # MW:   store the position and velocity COM
# MW_COM_p = MW_COM.COM_P(0.1)
# print(MW_COM_p)
# MW_COM_v = MW_COM.COM_V(MW_COM_p[0], MW_COM_p[1], MW_COM_p[2])
# print(MW_COM_v)


# ## 6 &nbsp; Testing Your Code

# ### 1.

# In[48]:


# MW CoM position and velocity vectors for the disk are already given above,
# so, let's print the results again:
# print(f'MW disk CoM position vector (t=0): {MW_COM_p}')
# print(f'MW disk CoM velocity vector (t=0): {MW_COM_v}')


# In[49]:


# # M31 (Andromeda) CoM position and velocity vectors:

# M31_COM = CenterOfMass("M31_000.txt", 2) # Create CenterOfMass object
# M31_COM_p = M31_COM.COM_P(0.1) # Get the CoM position with a tolerance of 0.1 kpc
# # Get the velocity of the COM (for particles within 15 kpc of the CoM)
# M31_COM_v = M31_COM.COM_V(M31_COM_p[0], M31_COM_p[1], M31_COM_p[2])

# # Print results
# print(f'M31 disk CoM position vector (t=0): {M31_COM_p}')
# print(f'M31 disk CoM velocity vector (t=0): {M31_COM_v}')


# In[50]:


# # M33 (Triangulum) CoM position and velocity vectors:

# M33_COM = CenterOfMass("M33_000.txt", 2) # Create CenterOfMass object
# M33_COM_p = M33_COM.COM_P(0.1) # Get the CoM position with a tolerance of 0.1 kpc
# # Get the velocity of the COM (for particles within 15 kpc of the CoM)
# M33_COM_v = M33_COM.COM_V(M33_COM_p[0], M33_COM_p[1], M33_COM_p[2])

# # Print results
# print(f'M33 disk CoM position vector (t=0): {M33_COM_p}')
# print(f'M33 disk CoM velocity vector (t=0): {M33_COM_v}')


# ### 2.

# In[51]:


# p_diff = MW_COM_p - M31_COM_p # displacement between centers
# sep = np.sqrt(p_diff[0]**2 + p_diff[1]**2 + p_diff[2]**2) # Distance formula
# print(f'Separation between MW and M31: {sep:.3f}') # print results


# In[52]:


# v_diff = MW_COM_v - M31_COM_v # relative velocity
# rel_v = np.sqrt(v_diff[0]**2 + v_diff[1]**2 + v_diff[2]**2) # magnitude
# print(f'Relative speed between MW and M31: {rel_v:.3f}') # print results


# **These values check out with roughly what we saw in Lecture 2.**

# ### 3.

# In[53]:


# p_diff2 = M31_COM_p - M33_COM_p # displacement between centers
# sep2 = np.sqrt(p_diff2[0]**2 + p_diff2[1]**2 + p_diff2[2]**2) # distance formula
# print(f'Separation between M31 and M33: {sep2:.3f}') # print results


# In[54]:


# v_diff2 = M31_COM_v - M33_COM_v # relative velocity
# rel_v2 = np.sqrt(v_diff2[0]**2 + v_diff2[1]**2 + v_diff2[2]**2) # magnitude
# print(f'Relative speed between M31 and M33: {rel_v2:.3f}') # print results


# ### 4.

# I figure that when these galaxies begin to merge, the centers of mass of *all* the stars that comprise them (or originally comprised them) become less important than where the cores of the original galaxies end up, i.e., where the central supermassive black holes/galactic nuclei are. As the galaxies begin to merge, the stars on the outskirts will start doing funky things and will get swapped between the two galaxies/begin to form the new super-galaxy. We don't want these to act as outliers that would skew our determination of where the centers of our original galaxies end up. The iterative process ensures that we get close enough to the centers of our original galaxies that the centers of mass converge to some values (within the specified tolerance), and we can reasonably treat those centers of mass as representing the new positions of the original galactic centers—though now only meaningfully representing the fraction of their original stars which remained within the critical radius for center-of-mass convergence. The other, exterior stars might as well be considered part of the new super-galaxy, methinks.
