#!/usr/bin/env python
# coding: utf-8

# # Evolution of Disk and Bulge S&eacute;rsic Profiles During the MW-M31 Major Merger

# My research project involves examing how the S&eacute;rsic profiles/S&eacute;rsic indices of the bulges and disks of the Milky Way and Andromeda (M31) galaxies evolve throughout their simulated future merger. This will be executed by fitting S&eacute;rsic profiles to snapshots of the radial intensity profile for each galaxy's bulge and disk.

# **Imports and Preferences:**

# In[1]:


# Load Modules
import numpy as np
import astropy.units as u

# import plotting modules
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import figure
import matplotlib.pyplot as plt

# plot preferences (font and resolution)
mpl.rcParams.update({
    'font.size': 9,
    'text.usetex': True,
    'font.family': 'Computer Modern Serif',
    'savefig.dpi': 1200
})

# my modules
from ReadFile import Read
from CenterOfMass import CenterOfMass
from MassProfile import MassProfile
from GalaxyMass import ComponentMass

# garbage collection
import gc

# fitting
from scipy.optimize import curve_fit

import sys

gc.collect()


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

gc.collect()


def sersic(r, R_e, n, M_tot):
    """ Function that computes a Sersic Profile assuming M/L = 1.
    
    PARMETERS
    ---------
        r : `float`
            Distance from the center of the galaxy (kpc)
            
        R_e : `float`
            Effective radius (2D radius that contains 
            half the light) (kpc)
            
        n :  `float`
            Sersic index
            
        M_tot : `float`
            Total stellar mass (Msun)

    RETURNS
    -------
        I: `array of floats`
            the radial intensity profile of the galaxy in Lsun/kpc^2

    """

    # We are assuming M/L = 1, so the total luminosity is:
    lum = M_tot
    
    # the effective intensity is
    I_e = lum / 7.2 / np.pi / R_e**2
    
    # Break down the equation 
    a = (r / R_e)**(1.0/n)
    b = -7.67 * (a-1)
    
    # Intensity
    #I = Ie*np.exp(-7.67*((r/R_e)**(1.0/n)-1.0))
    I = I_e * np.exp(b)
    
    del lum, I_e, a, b
    gc.collect()
    
    return I

gc.collect()


class RadialIntensity:
    """Creates an object that can compute and plot a radial intensity profile
    for a galaxy."""

    
    def __init__(self, galaxy, snap, res, comp, r_num, ax_type):
        """
        Initializes a RadialIntensity object, selecting the right data for the 
        given resolution (res), component (comp), and snapshot (snap) for galaxy
        galaxy. Plot colors, limits, and annotations are set based on the selections
        made for galaxy and comp. The filename to read in, the total mass of the
        given galaxy component, and the number of radii to convert into annuli are
        stored.
        
        Inputs:
            
            galaxy : `string`
                Name of the galaxy to create an intensity profile for ('MW' or 'M31')
                (case-sensitive)
                
            snap : `int`
                Snapshot (0–801) to read in.
                
            res : `string`
                'high' or 'low' simulation resolution (case-sensitive)
                
            comp : `string`
                Galaxy component, 'Disk' or 'Bulge' (case-sensitive)
                
            r_num : `int`
                numbr of radii to turn into annuli inner and outer radii for computing
                intensities within.
                
        Returns:
        
            None
        """
        
        # Select the desired simulation resolution
        if res == 'high':
            snap_path = 'HighRes_' + galaxy + '/'
        elif res == 'low':
            snap_path = 'VLowRes_' + galaxy + '/'
        
        # Component options
        if comp == 'Disk':
            self.p_type = 2 # particle type for disk particles
            
            # Differentiate colors by component _and_ galaxy
            if galaxy == 'MW':
                self.color = 'blue'
            else:
                self.color = 'red'
    
        elif comp == 'Bulge':
            self.p_type = 3 # particle type for bulge particles
            
            # Differentiate colors by component _and_ galaxy
            if galaxy == 'MW':
                self.color = 'orange'
            else:
                self.color='green'
            
        # store the component string for later use when plotting
        self.comp = comp
        
        # add a string of the filenumber to the value “000”
        snap_str= '000' + str(snap)
        # remove all but the last 3 digits
        snap_str = snap_str[-3:]
        
        # Store the snap_str for later use when writing plots to PNGs
        self.snap_str = snap_str

        # construct filename for later use
        data_path = '/Users/gabeweible/Library/CloudStorage/OneDrive-UniversityofArizona/junior/astr_400b/400B_2023_Weible/Data/'
        self.filename = data_path + snap_path + galaxy + '_' + snap_str + '.txt'
        
        # Store the galaxy for later use when plotting
        self.galaxy = galaxy
        # store the snap to compute the time in Gyr later
        self.snap = snap
        # store the number of radii for later annuli generation
        self.r_num = r_num
        
        # Store the total mass for the given component for creating a Sersic profile
        self.m_tot = ComponentMass(self.filename, self.p_type) * 1e12 # Msun
         
        # Create a center of mass object
        # This lets us get the x, y, z relative to the COM
        COM = CenterOfMass(self.filename, self.p_type)
        self.COM = COM
        
        COM_p = COM.COM_P(0.1) # COM position
        COM_v = COM.COM_V(*COM_p)
        
        # Save COM position and velocity
        self.COM_p = COM_p
        self.COM_v = COM_v
        
        
        del COM_p, COM_v, COM, r_num, snap, galaxy, data_path, comp
        gc.collect()
        
    
    def RotateFrame(self):
        """a function that will rotate the position and velocity vectors
        so that the disk angular momentum is aligned with z axis. 

        PARAMETERS
        ----------
            None

        RETURNS
        -------
            pos: `array of floats`
                rotated 3D array of positions (x,y,z) such that disk is in the XY plane
            vel: `array of floats`
                rotated 3D array of velocities (vx,vy,vz) such that disk angular momentum vector
                is in the +z direction 
        """

        # Create a COM of object for the Disk (always use the disk to rotate)
        COMD = CenterOfMass(self.filename, 2)
        
        # Compute COM using disk particles
        COMP = COMD.COM_P(0.1)
        COMV = COMD.COM_V(*COMP)
        
        # Determine positions of disk particles relative to COM 
        xD = COMD.x - COMP[0].value 
        yD = COMD.y - COMP[1].value 
        zD = COMD.z - COMP[2].value 

        # total magnitude
        rtot = np.sqrt(xD**2 + yD**2 + zD**2)

        # Determine velocities of disk particles relative to COM motion
        vxD = COMD.vx - COMV[0].value 
        vyD = COMD.vy - COMV[1].value 
        vzD = COMD.vz - COMV[2].value 

        # total velocity 
        vtot = np.sqrt(vxD**2 + vyD**2 + vzD**2)

        # Vectors for r and v 
        r = np.array([xD,yD,zD]).T # transposed 
        v = np.array([vxD,vyD,vzD]).T

        # compute the angular momentum
        L = np.sum(np.cross(r, v), axis=0)
        # normalize the vector
        L_norm = L/np.sqrt(np.sum(L**2))


        # Set up rotation matrix to map L_norm to z unit vector (disk in xy-plane)

        # z unit vector
        z_norm = np.array([0, 0, 1])

        # cross product between L and z
        vv = np.cross(L_norm, z_norm)
        s = np.sqrt(np.sum(vv**2))

        # dot product between L and z 
        c = np.dot(L_norm, z_norm)

        # rotation matrix
        I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        v_x = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
        R = I + v_x + np.dot(v_x, v_x)*(1 - c)/s**2
        
        # R will let us rotate any position or velocity into the Disk-rotated frame
        del I, v_x, c, s, vv, z_norm, L_norm, L, r, v, vtot, vxD, vyD, vzD, zD, yD, xD,
        COMP, COMV, COMD
        gc.collect()
        
        return R


    def profile(self, radii):
        """
        Calculates a radial area mass density profile using annuli in cylindrical
        coordinates.
        
        Inputs: 
        
            None
            
        Returns: 
            
            (r_annuli, I, R_e) : `tuple`
                Tuple of the annuli that area mass densities are calculated within,
                r_annuli, the intensity profile, I, and the effective
                (half-light / half-mass) radius, R_e. These are used for plotting.
        """
        
        # Get a vector to help us rotate
        R = self.RotateFrame()
        
        # selected particle positions and velocities wrt COM
        x_i = self.COM.x - self.COM_p[0].value
        y_i = self.COM.y - self.COM_p[1].value
        z_i = self.COM.z - self.COM_p[2].value
        m = self.COM.m
        
        # transposed to work with rotation
        r_i = np.array([x_i, y_i, z_i]).T
        
        # Rotated positions (not in COM frame)
        rr = np.dot(R, r_i.T).T
        
        # Extract components and use as before
        x = rr[:,0]
        y = rr[:,1]
        z = rr[:,2]

        # calculate the radial distances of particles in cylindrical coordinates
        cyl_r_mag = np.sqrt(x**2 + y**2)

        # create the mask to select particles enclosed for each radius
        # np.newaxis creates a virtual axis to make tmp_r_mag 2 dimensional
        # so that all radii can be compared simultaneously
        enc_mask = cyl_r_mag[:, np.newaxis] < np.asarray(radii).flatten()

        # calculate the enclosed masses 
        # relevant particles will be selected by enc_mask (i.e., *1)
        # outer particles will be ignored (i.e., *0)
        m_enc = np.sum(m[:, np.newaxis] * enc_mask, axis=0) * 1e10 # Msun

        # use the difference between nearby elements to get mass in each annulus
        m_annuli = np.diff(m_enc) # one element less then m_enc
        
        # Area mass density
        Sigma = m_annuli / (np.pi * (radii[1:]**2 - radii[:-1]**2))
        
        # convert area mass density to intensity with a mass-to-light ratio (1, here)
        m_to_l = 1
        I = Sigma / m_to_l

        r_annuli = np.sqrt(radii[1:] * radii[:-1]) 
        # here we choose the geometric mean
        
        half_mass = self.m_tot / 2
        indices = np.where(m_enc > half_mass)
        R_maj = radii[indices]

        # the first such index gives us the index of our half-light radius
        R_e = R_maj[0] # kpc

        # return the stuff that we need to make a plot
        del R_maj, indices, half_mass, m_to_l, Sigma, m_annuli, m_enc, enc_mask, radii,
        cyl_r_mag, z, y, x, rr, r_i, m, z_i, y_i, x_i, R
            
        gc.collect()
        
        return (r_annuli, I, R_e)
    
    
    def sersic_fit(self, r_annuli, I, R_e):
        """
        This method uses scipy.optimize.curve_fit to fit a Sersic profile (see the `sersic` function)
        given radii in r_annuli, the intensity profile I, equivalent radius R_e, and total mass
        M_tot.
        
        Inputs:
            
            r_annuli : `numpy array`
                1D array of float average radius for each annulus.
            
            I : `numpy array`
                1D array of float intensity within each annulus
                
            R_e : `float`
                Effective (half-light / half-mass radius) for the galaxy component
                
        Returns:
            
            popt : `numpy array`
                1D array of float parameters that will be passed to the `sersic` function
        """
        
        # use curve_fit to get the best parameter values in popt
        # Here we use a lambda function to only fit for n, and not R_e or M_tot
        popt, pcov = curve_fit(lambda r_annuli, n: sersic(r_annuli, R_e, n, self.m_tot), r_annuli, I)
        
        del pcov
        gc.collect()
        
        # Sersic index
        return popt[0] # change 1-element array to just a float
    
    
    def plot_profile(self, r_annuli, I, R_e, ax_type):
        """
        Uses Matplotlib to create PNGs of radial intensity profiles compared with 
        a de Vaucouleurs (Sersic index n = 4) profile.
        
        Inputs:
            
            r_annuli : `numpy array`
                1D array of float average radius for each annulus
            
            I : `numpy array`
                1D array of float intensity within each annulus
                
            R_e : `float`
                Effective (half-light / half-mass radius) for the galaxy component
                
            ax_type : `string`
                Linear-linear, log-log, or semilogy with 'linear', 'log', 'semilog'
                
        Returns:
            
            None
        """
        # fit the Sersic index n
        n = self.sersic_fit(r_annuli, I, R_e)
        
        # Simulation time
        Gyr = self.snap * 0.01 / 0.7
        
        if ax_type == 'log':
            # plot the fitted Sersic profile (log-log)
            ax.loglog(r_annuli, sersic(r_annuli, R_e, n, self.m_tot), color='k',
                         linestyle="-.", label=r'Fitted S\'{{e}}rsic $n={:.2f}$'.format(n),
                         linewidth=1)

            # Plot the calculated intensity profile (log-log)
            ax.loglog(r_annuli, I, alpha=0.8, label='Simulated '+self.comp,
                      linewidth=2, color=self.color)
            
            # Set the plot limits to something that works for them all
            ax.set(xlim=(10**(0), 0.4*10**(2)), ylim=(10**2, 10**11))
            
            # Time annotation
            plt.annotate(r'$\mathrm{{t}} = {:.2f} \ \mathrm{{Gyr}}$'.format(Gyr),
                     (2, 10**5), backgroundcolor='lightgrey', fontsize=10)
            
        elif ax_type == 'linear':
             # plot the fitted Sersic profile (log-log)
            ax.plot(r_annuli, sersic(r_annuli, R_e, n, self.m_tot), color='k',
                         linestyle="-.", label=r'Fitted S\'{{e}}rsic $n={:.2f}$'.format(n),
                         linewidth=1)

            # Plot the calculated intensity profile (log-log)
            ax.plot(r_annuli, I, alpha=0.8, label='Simulated '+self.comp,
                      linewidth=2, color=self.color)
            
            # axis limits
            ax.set(xlim=(0,40), ylim=(-0.1*10**8, 5*10**8))
            
            # Time annotation
            plt.annotate(r'$\mathrm{{t}} = {:.2f} \ \mathrm{{Gyr}}$'.format(Gyr),
                     (10, 2*10**8), backgroundcolor='lightgrey', fontsize=10)
            
        elif ax_type == 'semilog':
             # plot the fitted Sersic profile (log-log)
            ax.semilogy(r_annuli, sersic(r_annuli, R_e, n, self.m_tot), color='k',
                         linestyle="-.", label=r'Fitted S\'{{e}}rsic $n={:.2f}$'.format(n),
                         linewidth=1)

            # Plot the calculated intensity profile (log-log)
            ax.semilogy(r_annuli, I, alpha=0.8, label='Simulated '+self.comp,
                      linewidth=2, color=self.color)
            
            # axis limits
            ax.set(xlim=(0,40), ylim=(10**2, 10**11))
            
            # Time annotation
            plt.annotate(r'$\mathrm{{t}} = {:.2f} \ \mathrm{{Gyr}}$'.format(Gyr),
                     (10, 10**5), backgroundcolor='lightgrey', fontsize=10)
        
        # Axis labels, axis limits, and plot title
        ax.set(xlabel=r'$r \ (\mathrm{kpc})$',
               ylabel=r'$I \ \left(\mathrm{L_\odot} / \mathrm{kpc}^2\right)$',
               title=self.galaxy+' '+ self.comp + ' Particle Radial Intensity Profile')

        # legend and tidy things up
        ax.legend(loc='best')
        fig.tight_layout()
        
        # Set the folder to save the plot PNG to
        folder = galaxy + '_' + self.comp + '/'
        
        # Save the plot as a PNG
        plt.savefig(folder+ax_type+'/'+self.galaxy+'_'+self.snap_str+'_'+self.comp+'_'+'.png',
                    facecolor='w')
        
        # Clear the axes and figure
        plt.cla()
        ax.cla()
        fig.clf()
        fig.clear()
        plt.clf()
        
        # close everything
        plt.close('all')
        plt.close(fig)
        plt.close()
        plt.ioff()
        
        # delete variables
        del r_annuli, I, folder, Gyr, n
        
        # collect garbage
        gc.collect()

gc.collect()


# options
res = 'high' # resolution of data
r_num = 33 # number of radii to become (r_num - 1) annuli

# Choose which snaps to start and stop at, with what snap step
snap_start = 0 # inclusive
snap_end = 401 # exclusive
snap_step = 1

# Choose the galaxy and component
galaxy = 'MW'
component = 'Disk'

gc.collect()

# create a logspace (base 2) array of radii so that annuli radii increase
# as we move out into the more sparse outer regions
radii = np.logspace(np.log2(1), np.log2(40), num=r_num, base=2) # kpc

# Set up a plot with a 16:9 aspect ratio (UHD 4K)
fig = figure.Figure(figsize=(16/3, 9/3))
ax = fig.subplots()
        
ax_types = ['linear', 'log', 'semilog']
# Loop over axis plotting options
for ax_type in ax_types:

    # Loop over snapshots, get and plot surface brightness profiles using the SurfaceBrightness class
    for snap in range(snap_start, snap_end, snap_step):

        # Initialize the class
        radial_intensity = RadialIntensity(galaxy, snap, res, component, r_num, ax_type)

        # Get the radii, profile, and equivalent radius for the file used
        *plot_params ,= radial_intensity.profile(radii)

        # plot
        radial_intensity.plot_profile(*plot_params, ax_type)

        # delete variables
        del radial_intensity, plot_params
        # collect garbage
        gc.collect()

gc.collect()


# ## S&eacute;rsic Profiles
# 
# Used here to describe intensity (power per unit area) as a function of cylindrical radius for galaxies.

# ## ``sersic`` Function : 
# 
# We have a function called `sersic` that returns the S&eacute;rsic Profile in terms of the effective radius $R_\mathrm{e}$ (i.e. the half light radius).
# 
# $$\large I(r) = I_\mathrm{e} e^{-7.67 \left[ (r/R_\mathrm{e})^{1/n} - 1\right]} $$
# 
# Where 
# 
# $$\large L = 7.2\pi I_\mathrm{e} R_\mathrm{e}^2 $$
# 
# We will assume a mass to light ratio for disk and bulge particles of 1, so **this is also the half mass radius**, and so $\Sigma$, the projected area mass density, is nominally equivalent to the intensity $I$.
# 
# The function takes as input the radius, $R_e$, $n$ (S&eacute;rsic index) and the total stellar mass $M_\mathrm{tot}$ of the system.

# ## ``RadialIntensity`` Class:

# ## Options!

# ## Compute intensity profiles and plot

# Here we are plotting the calculated intensity profiles for our simulated galaxies as well as fitting S&eacute;rsic profiles by varying the S&eacute;rsic index $n$. The x axis is the radius within the galaxy componnnet and the y axis is the intensity calculated at that radius in cylindrical coordinates. Here the galaxy and component can be chosen above before looping through snapshots starting at ``snap_start``, ending at ``snap_end``, and with a step size of ``snap_step``. The goal of each plot is to show the S&eacute;rsic index at that snapshot, where $n=4$ is like an elliptical galaxy while $n=1$ or $n=2$ would be more characteristic of a disk. This tells us where the stars are radially, in particular how exactly they fall off in density with (cylindrical) radius.

# The above plots are completed, except for the fitting of S&eacute;rsic profiles, where now we just plot an $n=4$ template. 

# ## S&eacute;rsic index vs. time

# This plot will show how the best-fit S&eacute;rsic index varies with time through the simulation. Theoretically, the S&eacute;rsic indices should be increasing with time, at least for the disks which redistribute their stars to appear more elliptical. The x axis will be time and the y axis will be fitted S&eacute;rsic index. The particle type and galaxy will be chose above before looping over snapshots, which are chosen as explained above. This plot compiles the results from the previous plots for each snapshot, showing just how the fitted S&eacute;rsic indices change with time, which can then be shown in a single plot as opposed to having to select for individual snapshots or watching an animation. This is showing us how the distribution of stars is changing with time, where ellipticals are expected to have their light fall off more quickly than disks. This will help us to learn how the galaxies are transformed from disks to an elliptical galaxy during the simulation.

# ## Next steps:
# 2) See how the fitted S&eacute;rsic index changes with time and make a plot of that
# 3) Check out other cool effects (waves? Standing/traveling?)
