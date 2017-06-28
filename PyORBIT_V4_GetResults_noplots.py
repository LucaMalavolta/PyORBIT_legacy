from classes.model_container import ModelContainer
from classes.input_parser import yaml_parser
import classes.kepler_exo as kp
import numpy as np
import h5py
import cPickle as pickle
import scipy.optimize
import csv
import os
import argparse
import matplotlib as mpl
import sys
mpl.use('Agg')
from matplotlib import pyplot as plt
from classes import constants


def GelmanRubin(chains_T):
    # Courtesy of Luca "Sbuffo" Borsato
    n, M = np.shape(chains_T)

    theta_m = [np.mean(chains_T[:,i_m]) for i_m in range(0, M)]
    theta = np.mean(theta_m)

    d_theta2 = (theta_m - theta)**2
    B_n = np.sum(d_theta2) / (M-1)

    arg_W = [np.sum((chains_T[:,i_m] - theta_m[i_m])**2) / (n-1) for i_m in range(0, M)]
    W = np.mean(arg_W)

    n_frac = (n-1)/n
    var_plus = n_frac*W + B_n
    Var = var_plus + (B_n/M)

    Rc = np.sqrt(Var / W)
    return Rc


def get_mass(M_star2, M_star1, Period, K1, e0):
    # M_star1, M_star2 in solar masses
    # P in days -> Period is converted in seconds in the routine
    # inclination assumed to be 90 degrees
    # Gravitational constant is given in m^3 kg^-1 s^-2
    # output in m/s
    output = K1 - (2. * np.pi * G_grav * M_sun / 86400.0) ** (1.0 / 3.0) * (1.000 / np.sqrt(1.0 - e0 ** 2.0)) * (
                                                                                                                    Period) ** (
                                                                                                                    -1.0 / 3.0) * (
                      M_star2 * (M_star1 + M_star2) ** (-2.0 / 3.0))
    return output

G_grav = constants.Gsi # Gravitational Constants in SI system [m^3/kg/s^2]
G_ttvfast = constants.Giau  # G [AU^3/Msun/d^2]
M_SJratio = constants.Msjup
M_SEratio = constants.Msear
M_JEratio = constants.Mjear

R_SJratio = constants.Rsjup
R_JEratio = constants.Rjear
R_SEratio = constants.Rsjup * constants.Rjear

M_sun = constants.Msun
M_jup = constants.Mjup

Mu_sun = constants.Gsi * constants.Msun
seconds_in_day = constants.d2s
AU_km = constants.AU
AUday2ms = AU_km / seconds_in_day * 1000.0


parser = argparse.ArgumentParser(prog='PyORBIT_V3_GetResults.py', description='Extract results from output MCMC')
# parser.add_argument('-l', type=str, nargs='+', help='line identificator')
parser.add_argument('config_file', type=str, nargs=1, help='config file')
parser.add_argument('-p', type=str, nargs='?', default='False', help='Create plot files')
parser.add_argument('-mp', type=str, nargs='?', default='False', help='Create MEGA plot')
parser.add_argument('-v', type=str, nargs='?', default='False', help='Create Veusz ancillary files')
parser.add_argument('-t', type=str, nargs='?', default='False', help='Create GR traces')
parser.add_argument('-nburn', type=int, nargs='?', default=0, help='emcee burn-ins')
parser.add_argument('-c', type=str, nargs='?', default='False', help='Create Chains plots')
parser.add_argument('-forecast', type=float, nargs='?', default=None, help='Create Chains plots')

args = parser.parse_args()

sampler = 'emcee'
file_conf = args.config_file[0]

sample_keyword = {
    'polychord':['polychord', 'PolyChord', 'polychrod', 'poly'],
    'emcee': ['emcee', 'MCMC', 'Emcee']
}

# file_conf = raw_input()

mc = ModelContainer()
yaml_parser(file_conf, mc)

if sampler in sample_keyword['polychord'] and \
        mc.polychord_parameters['shutdown_jitter']:
    for dataset in mc.dataset_list:
        dataset.shutdown_jitter()

mc.initialize_model()

if bool(mc.pcv.dynamical):
    mc.dynamical_model.prepare(mc, mc.pcv)

M_star1 = mc.star_mass[0]
M_star1_err = mc.star_mass[1]


if sampler in sample_keyword['emcee']:

    dir_input = './' + mc.planet_name + '/emcee/'
    dir_output = './' + mc.planet_name + '/emcee_plot/'
    os.system('mkdir -p ' + dir_output)

    mc.variable_list = pickle.load(open(dir_input + 'vlist.pick', 'rb'))
    mc.scv.use_offset = pickle.load(open(dir_input + 'scv_offset.pick', 'rb'))

    print mc.variable_list

    h5f = h5py.File(dir_input + mc.planet_name + '.hdf5', "r")

    h5f_data = h5f['/data']
    h5f_emcee = h5f['/emcee']

    for item in h5f_emcee.attrs.keys():
        print item + ":", h5f_emcee.attrs[item]

        if item == 'nwalkers': mc.emcee_parameters['nwalkers']  = h5f_emcee.attrs[item]
        if item == 'ndim': mc.ndim = h5f_emcee.attrs[item]

    mc.bounds = h5f['/emcee/bound']
    chain = h5f['/emcee/chain']
    lnprobability = h5f['/emcee/lnprobability']
    acceptance_fraction = h5f['/emcee/acceptance_fraction']
    #acor = h5f['/emcee/acor']

    print mc.bounds[:]

    print
    print '*************************************************************'
    print
    print 'Acceptance Fraction for all walkers:'
    print acceptance_fraction[:]
    print
    print '*************************************************************'
    print

    if args.nburn > 0:
        mc.emcee_parameters['nburn'] = args.nburn

    if mc.emcee_parameters['nsave'] > 0:
        mc.emcee_parameters['nsteps'] = h5f_emcee.attrs['nsample']
        if mc.emcee_parameters['nburn'] > mc.emcee_parameters['nsteps']:
            mc.emcee_parameters['nburn'] = mc.emcee_parameters['nsteps'] / 4

    ntotal = np.int(mc.emcee_parameters['nsteps'] / mc.emcee_parameters['thin'])
    nburnin = np.int(mc.emcee_parameters['nburn'] / mc.emcee_parameters['thin'])

    lnprb_T = lnprobability[:][:].T

    chain_T = np.ndarray([ntotal, mc.emcee_parameters['nwalkers'], mc.ndim], dtype=np.double)
    for ii in xrange(0, mc.ndim):
        chain_T[:, :, ii] = chain[:, :, ii].T

    chain_burnt = chain_T[nburnin:, :, :]
    s = chain_burnt.shape
    lnprob_burnt = lnprb_T[nburnin:, :,]
    flatchain = chain_burnt.reshape(s[1] * s[0], s[2])
    flatlnprob = lnprob_burnt.reshape(s[1] * s[0])
    nsample = s[1] * s[0]
    #sel_flatchain = flatchain[:, 0] < 1.

    chain_med = np.asarray(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                               zip(*np.percentile(flatchain[:, :], [15.865, 50, 84.135], axis=0))))
    lnprob_med = np.percentile(flatlnprob, [15.865, 50, 84.135], axis=0)
    lnprob_med[1:] = np.abs(lnprob_med[1:]-lnprob_med[0])

    mc.results_resumen(chain_med[:, 0])

    print
    print '*************************************************************'
    print

    lnprob_median = np.median(flatlnprob)

    print 'LNprob median = ', lnprob_median

    print
    print 'Reference Time Tref: ', mc.Tref
    print
    print '*************************************************************'
    print


if sampler in sample_keyword['polychord']:

    dir_input = './' + mc.planet_name + '/' + mc.polychord_parameters['base_dir']
    dir_output = './' + mc.planet_name + '/polychord_plot/'
    os.system('mkdir -p ' + dir_output)

    data_in = np.genfromtxt(dir_input+mc.planet_name+'_equal_weights.txt')
    flatlnprob = data_in[:, 1]
    flatchain = data_in[:, 2:]
    nsample = np.size(flatlnprob)

    print
    print '*************************************************************'
    print

    chain_med = np.asarray(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                               zip(*np.percentile(flatchain[:, :], [15.865, 50, 84.135], axis=0))))
    lnprob_med = np.percentile(flatlnprob, [15.865, 50, 84.135], axis=0)
    lnprob_med[1:] = np.abs(lnprob_med[1:]-lnprob_med[0])
    mc.results_resumen(chain_med[:, 0])

    print
    print '*************************************************************'
    print

    lnprob_median = np.median(flatlnprob)
    print 'LNprob median = ', lnprob_median

    print
    print 'Tref: ', mc.Tref
    print
    print '*************************************************************'
    print



x0 = 1. / 150

M_star1_rand = np.random.normal(M_star1, M_star1_err, nsample)

""" Creation of the directory for the plots"""
plot_dir = dir_output + '/files_plot/'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

boundaries = np.asarray([mc.Tref, mc.Tref])
plot_dir = dir_output + '/files_plot/'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
for dataset in mc.dataset_list:
    if dataset.kind == 'RV':
        boundaries[0] = min(boundaries[0], dataset.x[0])
        boundaries[1] = max(boundaries[1], dataset.x[-1])
# tenpercent = (boundaries[1] - boundaries[0]) / 10.
# boundaries = [boundaries[0] - tenpercent, boundaries[1] + tenpercent]
boundaries += np.asarray([-1.0, 1.0]) * (boundaries[1] - boundaries[0]) / 10.

x_range_step = max(0.01, (boundaries[1] - boundaries[0]) / 100000)
x_range = np.arange(boundaries[0], boundaries[1], x_range_step)
x_phase = np.arange(-0.50, 1.50, 0.005, dtype=np.double)

model_dsys, model_plan, model_orbs, model_actv, model_curv = mc.rv_make_model(chain_med[:, 0], x_range, x_phase)

"""Datasets summary"""


if 'kepler' in mc.model_list:

    if sampler in sample_keyword['polychord']:
        ''' Special plot for the polychord case:
            Let's save all the sample together
        '''
        sample_total = {}

    #if args.p != 'False' or args.v != 'False':

    for planet_name in mc.pcv.planet_name:

        print 'Planet ', planet_name, ' summary'
        print

        dynamical_flag = (planet_name in mc.pcv.dynamical)


        """ Let's feed the conversion function with the average results to get out a list of human variables
            Before doing that, the 1-sigma intervals must be converted to the actual values
        """
        chain_sig = chain_med[:, 0]

        convert_med = mc.pcv.convert(planet_name, chain_sig)

        n_orbital = len(mc.pcv.var_list[planet_name])
        n_fitted = len(mc.pcv.var_list[planet_name]) - len(mc.pcv.fix_list[planet_name])

        n_curv = mc.ccv.order

        sample_plan = np.zeros([nsample, n_orbital+6+n_curv])
        median_tmp  = np.zeros([n_orbital+6+n_curv])


        """Let's put all the human variables - including those that have been fixed - in sample_plan
           An index is assigned to each variable to keep track of them in
           We copy the median and sigma values from the derived distribution"""
        convert_out = {}
        for n_var, var in enumerate(convert_med):
            convert_out[var] = n_var
            median_tmp[n_var] = convert_med[var]


        for ii in xrange(0, nsample):
            convert_tmp = mc.pcv.convert(planet_name, flatchain[ii, :])
            for var in convert_out:
                sample_plan[ii, convert_out[var]] = convert_tmp[var]

        convert_out['Tperi'] = n_orbital + 1
        convert_out['Tcent'] = n_orbital + 2
        convert_out['M_kep'] = n_orbital + 3
        convert_out['a_smj'] = n_orbital + 4

        # Time of Periastron
        sample_plan[:, convert_out['Tperi']] = mc.Tref + (-sample_plan[:, convert_out['f']] + sample_plan[:, convert_out['o']]) / \
            (2*np.pi) * sample_plan[:, convert_out['P']]

        # Time of transit center
        sample_plan[:, convert_out['Tcent']] = mc.Tref + kp.kepler_Tcent_T0P(
            sample_plan[:, convert_out['P']], sample_plan[:, convert_out['f']],
            sample_plan[:, convert_out['e']], sample_plan[:, convert_out['o']])

        if dynamical_flag:
            convert_out['K'] = n_orbital + 5
            sample_plan[:, convert_out['K']] = kp.kepler_K1(mc.star_mass[0],
                                                            sample_plan[:, convert_out['M']]/mc.M_SEratio,
                                                            sample_plan[:, convert_out['P']],
                                                            sample_plan[:, convert_out['i']],
                                                            sample_plan[:, convert_out['e']])

        if 'curvature' in mc.model_list:
            for n_var, var in enumerate(mc.ccv.list_pams):
                convert_out[var] = n_orbital + 6 + n_var

            for ii in xrange(0, nsample):
                convert_tmp = mc.ccv.convert(flatchain[ii, :])
                for var in mc.ccv.list_pams:
                    sample_plan[ii, convert_out[var]] = convert_tmp[var]

            #for var in mc.ccv.list_pams:
            #    sample_med[convert_out[var], 0] = np.median(sample_plan[:, convert_out[var]])

        for n, (P, K, e, M) in enumerate(zip(sample_plan[:, convert_out['P']],
                                             sample_plan[:, convert_out['K']],
                                             sample_plan[:, convert_out['e']],
                                             M_star1_rand)):
            # Planet mass
            sample_plan[n, convert_out['M_kep']] = mc.M_SJratio * scipy.optimize.fsolve(get_mass, x0, args=(M, P, K, e))
            # semi-major axis
            sample_plan[n, convert_out['a_smj']] = np.power(
                (Mu_sun * np.power(P * seconds_in_day / (2 * np.pi), 2) / (AU_km ** 3.0)) * M, 1.00 / 3.00)

        if not dynamical_flag:
            if mc.pcv.inclination[planet_name][1] > 0.01:
                sample_plan[:, convert_out['M_kep']] = sample_plan[:, convert_out['M_kep']] / np.sin(np.pi / 180. *
                        np.random.normal(mc.pcv.inclination[planet_name][0], mc.pcv.inclination[planet_name][1], nsample))

        """ Rescale the center point and range for omega and phase"""
        for key in ['o','f']:
            var_index = convert_out[key]
            sel_turnA = (sample_plan[:, var_index] > np.pi + median_tmp[var_index])
            sel_turnB = (sample_plan[:, var_index] < median_tmp[var_index] - np.pi)
            sample_plan[sel_turnA, var_index] -= 2*np.pi
            sample_plan[sel_turnB, var_index] += 2*np.pi

        sample_med = np.asarray(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                    zip(*np.percentile(sample_plan[:, :], [15.865, 50, 84.135], axis=0))))

        #sample_med[o_index, 0] = sample_tmp[o_index,0]
        #sample_plan_tmp = sample_plan[o_index, :]

        #sample_med[o_index, 1:] = np.abs(np.percentile(sample_plan[o_index, :], [15.865, 84.135]))-sample_med[o_index, 0]

        e_med = np.percentile(sample_plan[:, convert_out['e']], 65.865, axis=0)

        if sampler in sample_keyword['polychord']:
            ''' Let's save all the samples for this specific planet, to be used later in the special PolyChrod plots'''
            sample_total[planet_name] = {'sample_plan': sample_plan, 'convert_out': convert_out}

        print 'Period = ', sample_med[convert_out['P'], 0], ' +\sigma ', sample_med[convert_out['P'], 1], ' -\sigma ', sample_med[convert_out['P'], 2]
        print 'K      = ', sample_med[convert_out['K'], 0], ' +\sigma ', sample_med[convert_out['K'], 1], ' -\sigma ', sample_med[convert_out['K'], 2]
        print 'phase  = ', sample_med[convert_out['f'], 0], ' +\sigma ', sample_med[convert_out['f'], 1], ' -\sigma ', sample_med[convert_out['f'], 2]
        print 'e      = ', sample_med[convert_out['e'], 0], ' +\sigma ', sample_med[convert_out['e'], 1], ' -\sigma ', sample_med[convert_out['e'], 2], ', < ', e_med
        print 'o      = ', sample_med[convert_out['o'], 0], ' +\sigma ', sample_med[convert_out['o'], 1], ' -\sigma ', sample_med[convert_out['o'], 2]

        if dynamical_flag:
            print 'lN     = ', sample_med[convert_out['lN'], 0], ' +\sigma ', sample_med[convert_out['lN'], 1], ' -\sigma ', sample_med[convert_out['lN'], 2]
            print 'i      = ', sample_med[convert_out['i'], 0], ' +\sigma ', sample_med[convert_out['i'], 1], ' -\sigma ', sample_med[convert_out['i'], 2]
            print 'Mass_J = ', sample_med[convert_out['M'], 0]/mc.M_JEratio, ' +\sigma ', sample_med[convert_out['M'], 1]/mc.M_JEratio, ' -\sigma ', sample_med[convert_out['M'], 2]/mc.M_JEratio
            print 'Mass_E = ', sample_med[convert_out['M'], 0], ' +\sigma ', sample_med[convert_out['M'], 1], ' -\sigma ', sample_med[convert_out['M'], 2]
        else:
            print 'Mass_J = ', sample_med[convert_out['M_kep'], 0], ' +\sigma ', sample_med[convert_out['M_kep'], 1], ' -\sigma ', sample_med[convert_out['M_kep'], 2]
            print 'Mass_E = ', sample_med[convert_out['M_kep'], 0]*mc.M_JEratio, ' +\sigma ',sample_med[convert_out['M_kep'], 1]*mc.M_JEratio, ' -\sigma ',sample_med[convert_out['M_kep'], 2]*mc.M_JEratio

        print 'Tperi  = ', sample_med[convert_out['Tperi'], 0], ' +\sigma ', sample_med[convert_out['Tperi'], 1], ' -\sigma ', sample_med[convert_out['Tperi'], 2]
        print 'Tcent  = ', sample_med[convert_out['Tcent'], 0], ' +\sigma ', sample_med[convert_out['Tcent'], 1], ' -\sigma ', sample_med[convert_out['Tcent'], 2]
        print 'a      = ', sample_med[convert_out['a_smj'], 0], ' +\sigma ', sample_med[convert_out['a_smj'], 1], ' -\sigma ', sample_med[convert_out['a_smj'], 2]
        print
        print 'Planet ', planet_name, ' completed'
        print
        print '-----------------------------'
        print


if 'curvature' in mc.model_list:

    print 'Curvature summary'
    print

    n_vars = 0
    sample_plan_transpose = []
    sel_label = []

    for name in mc.ccv.list_pams:
        n_vars += 1
        var = flatchain[:, mc.ccv.var_list[name]]
        var_phys = mc.ccv.variables[name](var, var, xrange(0, nsample))
        sample_plan_transpose.append(var_phys)
        sel_label.append(name)

    sample_plan = np.asarray(sample_plan_transpose).T
    sample_med = np.asarray(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                    zip(*np.percentile(sample_plan[:, :], [15.865, 50, 84.135], axis=0))))

    for lab, sam in zip(sel_label, sample_med):
        print lab,' = ', sam[0], ' +\sigma ', sam[1], ' -\sigma ', sam[2]
    print

    print 'Curvature summary completed'
    print
    print '-----------------------------'
    print

print
