from PyORBIT_V2_Classes import *
import numpy as np
import emcee
from pyde.de import DiffEvol
import h5py
import cPickle as pickle
import os
#import json

file_conf  = raw_input()

mc = ModelContainer()

get_pyorbit_input(file_conf, mc)

dir_output = './' + mc.planet_name + '/'
if not os.path.exists(dir_output):
    os.makedirs(dir_output)

mc.create_bounds()
print 'Dimensions = ', mc.ndim

mc.nwalkers = mc.ndim * mc.npop_mult
if mc.nwalkers%2 == 1: mc.nwalkers += 1

print 'Nwalkers = ', mc.nwalkers

if os.path.isfile(dir_output + 'pyde_pops.pick'):
        population = pickle.load(open(dir_output + 'pyde_pops.pick', 'rb'))
        pyde_mean = pickle.load(open(dir_output + 'pyde_mean.pick', 'rb'))
else:
    print 'PyDE'
    de = DiffEvol(mc, mc.bounds, mc.nwalkers, maximize=True)
    de.optimize(mc.ngen)
    print 'PyDE completed'

    population = de.population
    pyde_mean = np.mean(population, axis=0)
    pickle.dump(pyde_mean, open(dir_output + 'pyde_mean.pick', 'wb'))

    #np.savetxt(dir_output + 'pyDEout_original_bounds.dat', mc.bounds)
    #np.savetxt(dir_output + 'pyDEout_original_pops.dat', population)

    # bounds redefinition and fix for PyDE anomalous results
    if mc.recenter_bounds_flag:
        pickle.dump(mc.bounds, open(dir_output + 'bounds_orig.pick', 'wb'))
        pickle.dump(population, open(dir_output + 'pyde_pops_orig.pick', 'wb'))
        mc.recenter_bounds(pyde_mean, population)
        pickle.dump(mc.bounds, open(dir_output + 'bounds.pick', 'wb'))
        pickle.dump(population, open(dir_output + 'pyde_pops.pick', 'wb'))

        #np.savetxt(dir_output + 'pyDEout_redefined_bounds.dat', mc.bounds)
        #np.savetxt(dir_output + 'pyDEout_redefined_pops.dat', de.population)
        print 'REDEFINED BOUNDS'

    else:
        pickle.dump(mc.bounds, open(dir_output + 'bounds.pick', 'wb'))
        pickle.dump(population, open(dir_output + 'pyde_pops.pick', 'wb'))

mc.results_resumen(pyde_mean)

#json.dump(mc.variable_list, open('output/' + mc.planet_name + '_vlist.json', 'wb'))
pickle.dump(mc.variable_list, open(dir_output + 'vlist.pick', 'wb'))
pickle.dump(mc.scv.use_offset,  open(dir_output + 'scv_offset.pick', 'wb'))

print 'emcee'
sampler = emcee.EnsembleSampler(mc.nwalkers, mc.ndim, mc, threads=24)
sampler.run_mcmc(population, mc.nsteps, thin=mc.thin)

print 'emcee completed'

# json.dump(mc.variable_list, open('output/' + mc.planet_name + '_vlist.json', 'wb'))
# pickle.dump(mc.variable_list, open(dir_output + 'vlist.pick', 'wb'))
# pickle.dump(mc.scv.use_offset,  open(dir_output + 'scv_offset.pick', 'wb'))

h5f = h5py.File(dir_output + mc.planet_name + '.hdf5', "w")

data_grp = h5f.create_group("data")
data_grp.attrs.create('file_conf',data=file_conf)

data_grp.create_dataset("pyDE_mean", data=pyde_mean, compression="gzip")
data_grp.create_dataset("pyDE_pops", data=de.population, compression="gzip")

emcee_grp = h5f.create_group("emcee")
emcee_grp.attrs.create("nwalkers", data=mc.nwalkers)
emcee_grp.attrs.create("ndim", data=mc.ndim)


emcee_grp.create_dataset("bound", data=mc.bounds, compression="gzip")
emcee_grp.create_dataset("chain", data=sampler.chain, compression="gzip")

emcee_grp.create_dataset("lnprobability", data=sampler.lnprobability, compression="gzip")
emcee_grp.create_dataset("acceptance_fraction", data=sampler.acceptance_fraction, compression="gzip")
emcee_grp.create_dataset("acor", data=sampler.acor, compression="gzip")

h5f.close()

