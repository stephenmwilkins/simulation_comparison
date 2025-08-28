
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import cmasher as cmr
from astropy.cosmology import Planck15 as cosmo

from unyt import sr, arcmin
from scipy import interpolate

plt.style.use('matplotlibrc.txt')


colour_by_redshift_endpoint = False

# add surveys
add_surveys = True
survey_redshift = 7.0

# force numbering of all simulations
number_all = True

# initialise figure
fig = plt.figure(figsize=(4, 4))
left = 0.125
bottom = 0.125
height = 0.85
width = 0.85
ax = fig.add_axes((left, bottom, width, height))

# set axes limits
ax.set_xlim([12.5, 3.])
ax.set_ylim([5.5, 12.99])

# set colour scale for redshift end-point
norm = mpl.colors.Normalize(vmin=0, vmax=10)
cmap = cmr.cosmic


# add constant particle number
# for i in range(15):
#     ax.plot([11, 2], [4+i, -3+i], lw = 1, c='k', alpha = 0.025, zorder = -1)

simulations = {}

# Millennium
h = 0.7
simulations['Millennium'] = {'size': 500/h, 'm_dm': 8.6E8/h, 'complete': True, 'redshift_end': 0.0, 'label': False}

# Millennium-II
h = 0.7
simulations['Millennium-II'] = {'size': 100/h, 'm_dm': 6.89E6/h, 'complete': True, 'redshift_end': 0.0, 'label': False}

# P-Millennium
h = 0.7
simulations['P-Millennium'] = {'size': 542/h, 'm_dm': 1.06E8/h, 'complete': True, 'redshift_end': 0.0, 'label': False}

# FLAMINGO
simulations['FLAMINGO-L1_m8_DMO'] = {'size': 1000, 'm_dm': 8.4E8, 'complete': True, 'redshift_end': 0.0, 'label': False}
simulations['FLAMINGO-L1_m9_DMO'] = {'size': 1000, 'm_dm': 6.72E9, 'complete': True, 'redshift_end': 0.0, 'label': False}
simulations['FLAMINGO-L1_m10_DMO'] = {'size': 1000, 'm_dm': 5.38E10, 'complete': True, 'redshift_end': 0.0, 'label': False}
simulations['FLAMINGO-L2p8_m9_DMO'] = {'size': 2800, 'm_dm': 6.72E9, 'complete': True, 'redshift_end': 0.0, 'label': False}
simulations['FLAMINGO-L5p6_m10_DMO'] = {'size': 5600, 'm_dm': 5.38E10, 'complete': True, 'redshift_end': 0.0, 'label': False}
simulations['FLAMINGO-L11p2_m11_DMO'] = {'size': 11200, 'm_dm': 4.3E11, 'complete': True, 'redshift_end': 0.0, 'label': False}
simulations['FLAMINGO-10K'] = {'size': 2800, 'm_dm': 8.4E8, 'complete': True, 'redshift_end': 0.0, 'label': False}



# surveys
surveys = {}

# All sky
surveys['All Sky'] = {'area': (4*np.pi*sr).to('arcmin**2').value}

# Euclid
surveys['Euclid/Deep'] = {'area': 40*3600}
surveys['Euclid/Wide'] = {'area': 18000*3600}

# Webb
surveys['Webb/COSMOS-Web'] = {'area': 0.6*3600}
# surveys['Webb/NGDEEP'] = {'area': 8.}






# list of simulations labelled with a number
labels = []

# starting index for simulations labelled with a number 
j = 1

for i, (simulation_name, simulation) in enumerate(simulations.items()):

    s = simulation

    marker = 'o'

    # set transparency based on whether completed or not
    if s['complete']:
        alpha = 1.0
    else:
        alpha = 0.3

    # set colour 
    if colour_by_redshift_endpoint:
        c = cmap(norm(s['redshift_end']))
    else:
        c = 'k'

    ax.scatter(
        np.log10(s['m_dm']),
        3*np.log10(s['size']),
        c=[c],
        s=20,
        lw=0,
        marker=marker,
        zorder=2,
        alpha=alpha)

    # if the point should be immediately labelled put it here.
    if (s['label'] is True) and (number_all is False):

        ax.text(
            np.log10(s['m_dm']),
            3*np.log10(s['size'])-0.25,
            simulation_name,
            fontsize=7,
            ha='center',
            va='center',
            alpha=alpha,
            c=c)

    # if not add it to a list to label later
    else:

        # add number label
        ax.text(
            np.log10(s['m_dm'])+0.075,
            3*np.log10(s['size'])-0.15,
            j,
            fontsize=7,
            ha='right',
            va='center',
            alpha=alpha,
            c=c)

        # add to list for annotating later
        labels.append(simulation_name)

        # increment label
        j += 1

# add labels for numbered simulations
x_pos = 12.
y_start_pos = 7.8
y_increment = 0.2

for i, simulation in enumerate(labels):

    label = rf'{i+1}: {simulation}'

    ax.text(
        x_pos,
        y_start_pos - i*y_increment,
        label,
        c='k',
        fontsize=6,
        ha='left',
        va='bottom')



# add surveys as horizontal lines
if add_surveys:

    v1, v2 = cosmo.comoving_volume([survey_redshift-0.5, survey_redshift+0.5]).to('Mpc3')

    for survey_name, survey in surveys.items():

        area = survey['area']

        volume = -np.log10(1. / ((v2 - v1) * (area/(41253.*3600))).value)

        print(survey_name, volume, 10**volume)

        ax.axhline(volume, c='k', alpha=0.05, lw=2)
        ax.text(
            4.1,
            volume+0.15,
            survey_name,
            size=7,
            va='center',
            ha='right',
            color='k',
            zorder=4,
            alpha=0.5)


# add axes labels
ax.set_xlabel(r'$\rm\log_{10}(DM\ resolution\ element\ mass/M_{\odot})$')
ax.set_ylabel(r'$\rm\log_{10}(volume/cMpc^{3})$')

fig.savefig(f'figs/dm_volume.pdf')
fig.savefig(f'figs/dm_volume.png')
fig.clf()
