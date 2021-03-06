import h5py
import numpy as np
import kepler_exo as kp
from matplotlib import pyplot as plt
import corner as triangle
#from scipy.interpolate import *

print 'Waiting for .config file name...'

file_conf = raw_input()
#file_conf = 'OC101_fast.config'
file=open(file_conf,'r')
row = file.readlines()

bin_num = 20
print 'bin_num: ',bin_num

for line in row:
  if line.find("Output") > -1:
    info = line.split()
    planet_name = info[1]
  if line.find("Nburn") > -1:
    info = line.split()
    nburn = np.asarray(info[1], dtype=np.int64)

h5f = h5py.File('output/'+planet_name+'.hdf5', "r")

for item in h5f.attrs.keys():
  print item + ":", h5f.attrs[item]
  print item + ":", h5f.attrs[item]
h5f_data  = h5f['/data']
h5f_emcee = h5f['/emcee']
for item in h5f_data.attrs.keys():
  print item + ":", h5f_data.attrs[item]

  if item == 'flag_rv': flag_rv = h5f_data.attrs[item]
  if item == 'flag_p1': flag_p1 = h5f_data.attrs[item]
  if item == 'flag_p2': flag_p2 = h5f_data.attrs[item]
  if item == 'flag_fw': flag_fw = h5f_data.attrs[item]
  if item == 'flag_bs': flag_bs = h5f_data.attrs[item]
  if item == 'Tref'   : Tref    = h5f_data.attrs[item]
  if item == 'rv_n' : rv_n = h5f_data.attrs[item]
  if item == 'p1_n' : p1_n = h5f_data.attrs[item]
  if item == 'p2_n' : p2_n = h5f_data.attrs[item]
  if item == 'fw_n' : fw_n = h5f_data.attrs[item]
  if item == 'bs_n' : bs_n = h5f_data.attrs[item]

  if item == 'n_rvj' : n_rvj = h5f_data.attrs[item]
  if item == 'n_rvo' : n_rvo = h5f_data.attrs[item]
  if item == 'n_rvl' : n_rvl = h5f_data.attrs[item]
  if item == 'n_rva' : n_rva = h5f_data.attrs[item]
  if item == 'n_p1j' : n_p1j = h5f_data.attrs[item]
  if item == 'n_p1o' : n_p1o = h5f_data.attrs[item]
  if item == 'n_p1l' : n_p1l = h5f_data.attrs[item]
  if item == 'n_p2j' : n_p2j = h5f_data.attrs[item]
  if item == 'n_p2o' : n_p2o = h5f_data.attrs[item]
  if item == 'n_p2l' : n_p2l = h5f_data.attrs[item]
  if item == 'n_fwj' : n_fwj = h5f_data.attrs[item]
  if item == 'n_fwo' : n_fwo = h5f_data.attrs[item]
  if item == 'n_fwl' : n_fwl = h5f_data.attrs[item]
  if item == 'n_fwa' : n_fwa = h5f_data.attrs[item]
  if item == 'n_bsj' : n_bsj = h5f_data.attrs[item]
  if item == 'n_bso' : n_bso = h5f_data.attrs[item]
  if item == 'n_bsl' : n_bsl = h5f_data.attrs[item]
  if item == 'n_bsa' : n_bsa = h5f_data.attrs[item]

for item in h5f_emcee.attrs.keys():
  print item + ":", h5f_emcee.attrs[item]
  if item == 'kind_sin'          : kind_sin  = h5f_emcee.attrs[item]
  if item == 'n_planets'         : n_planets = h5f_emcee.attrs[item]
  if item == 'n_periods'         : n_periods = h5f_emcee.attrs[item]
  if item == 'n_trends'          : n_trends = h5f_emcee.attrs[item]
  if item == 'n_orbpams'         : n_orbpams = h5f_emcee.attrs[item]
  if item == 'n_syspams'         : n_syspams = h5f_emcee.attrs[item]
  if item == 'n_sinpams'         : n_sinpams = h5f_emcee.attrs[item]
  if item == 'n_rotpams'         : n_rotpams = h5f_emcee.attrs[item]
  if item == 'n_extpams'         : n_extpams = h5f_emcee.attrs[item]
  if item == 'n_dataset'          : n_dataset  = h5f_emcee.attrs[item]
  if item == 'n_pin'             : n_pin     = h5f_emcee.attrs[item]
  if item == 'n_amp'             : n_amp     = h5f_emcee.attrs[item]
  if item == 'n_pha'             : n_pha     = h5f_emcee.attrs[item]
  if item == 'n_pof'             : n_pof     = h5f_emcee.attrs[item]
  if item == 'ndim'              : ndim      = h5f_emcee.attrs[item]
  if item == 'nsteps'            : nsteps    = h5f_emcee.attrs[item]
  if item == 'ngen'              : ngen      = h5f_emcee.attrs[item]
  if item == 'npop'              : npop      = h5f_emcee.attrs[item]
  #if item == 'nburn'             : nburn     = h5f_emcee.attrs[item]
  if item == 'thin'              : thin      = h5f_emcee.attrs[item]


nburn = nburn/thin
nsteps = nsteps/thin

n_pho = np.amax(n_amp)



rvp_flag = np.asarray(h5f['/data/rvp_flag'], dtype=bool)
p1p_flag = np.asarray(h5f['/data/p1p_flag'], dtype=bool)
p2p_flag = np.asarray(h5f['/data/p2p_flag'], dtype=bool)
fwp_flag = np.asarray(h5f['/data/fwp_flag'], dtype=bool)
bsp_flag = np.asarray(h5f['/data/bsp_flag'], dtype=bool)

if flag_rv:
  data_rv  = h5f['/data/data_rv']
  rv_x = np.asarray(data_rv[:,0], dtype=np.double)
  rv_y = np.asarray(data_rv[:,1], dtype=np.double)
  rv_e = np.asarray(data_rv[:,2], dtype=np.double)
  rv_j = np.asarray(data_rv[:,3], dtype=np.double)
  rv_o = np.asarray(data_rv[:,4], dtype=np.double)
  rv_l = np.asarray(data_rv[:,5], dtype=np.double)

  rvj_mask = np.asarray(h5f['/data/rvj_mask'], dtype=np.double)
  rvo_mask = np.asarray(h5f['/data/rvo_mask'], dtype=np.double)
  rvl_mask = np.asarray(h5f['/data/rvl_mask'], dtype=np.double)
  rva_mask = np.asarray(h5f['/data/rva_mask'], dtype=np.double)

## Reading photometric data, if present:
if flag_p1:
  data_p1  = h5f['/data/data_p1']
  p1_x = np.asarray(data_p1[:,0], dtype=np.double)
  p1_y = np.asarray(data_p1[:,1], dtype=np.double)
  p1_e = np.asarray(data_p1[:,2], dtype=np.double)
  p1_j = np.asarray(data_p1[:,3], dtype=np.double)
  p1_o = np.asarray(data_p1[:,4], dtype=np.double)
  p1_l = np.asarray(data_p1[:,5], dtype=np.double)

  p1j_mask = np.asarray(h5f['/data/p1j_mask'], dtype=np.double)
  p1o_mask = np.asarray(h5f['/data/p1o_mask'], dtype=np.double)
  p1l_mask = np.asarray(h5f['/data/p1l_mask'], dtype=np.double)

# photometric data in a different filter
if flag_p2:
  data_p2  = h5f['/data/data_p2']
  p2_x = np.asarray(data_p2[:,0], dtype=np.double)
  p2_y = np.asarray(data_p2[:,1], dtype=np.double)
  p2_e = np.asarray(data_p2[:,2], dtype=np.double)
  p2_j = np.asarray(data_p2[:,3], dtype=np.double)
  p2_o = np.asarray(data_p2[:,4], dtype=np.double)
  p2_l = np.asarray(data_p2[:,5], dtype=np.double)

  p2j_mask = np.asarray(h5f['/data/p2j_mask'], dtype=np.double)
  p2o_mask = np.asarray(h5f['/data/p2o_mask'], dtype=np.double)
  p2l_mask = np.asarray(h5f['/data/p2l_mask'], dtype=np.double)
## Reading FWHM data, if present:
if flag_fw:
  data_fw  = h5f['/data/data_fw']
  fw_x = np.asarray(data_fw[:,0], dtype=np.double)
  fw_y = np.asarray(data_fw[:,1], dtype=np.double)
  fw_e = np.asarray(data_fw[:,2], dtype=np.double)
  fw_j = np.asarray(data_fw[:,3], dtype=np.double)
  fw_o = np.asarray(data_fw[:,4], dtype=np.double)
  fw_l = np.asarray(data_fw[:,5], dtype=np.double)

  fwj_mask = np.asarray(h5f['/data/fwj_mask'], dtype=np.double)
  fwo_mask = np.asarray(h5f['/data/fwo_mask'], dtype=np.double)
  fwl_mask = np.asarray(h5f['/data/fwl_mask'], dtype=np.double)
  fwa_mask = np.asarray(h5f['/data/fwa_mask'], dtype=np.double)

## Reading BIS data, if present:
if flag_bs:
  data_bs  = h5f['/data/data_bs']
  bs_x = np.asarray(data_bs[:,0], dtype=np.double)
  bs_y = np.asarray(data_bs[:,1], dtype=np.double)
  bs_e = np.asarray(data_bs[:,2], dtype=np.double)
  bs_j = np.asarray(data_bs[:,3], dtype=np.double)
  bs_o = np.asarray(data_bs[:,4], dtype=np.double)
  bs_l = np.asarray(data_bs[:,5], dtype=np.double)

  bsj_mask = np.asarray(h5f['/data/bsj_mask'], dtype=np.double)
  bso_mask = np.asarray(h5f['/data/bso_mask'], dtype=np.double)
  bsl_mask = np.asarray(h5f['/data/bsl_mask'], dtype=np.double)
  bsa_mask = np.asarray(h5f['/data/bsa_mask'], dtype=np.double)

bounds = h5f['/emcee/bound']
chain  = h5f['/emcee/chain']
blobs  = h5f['/emcee/blobs']
#blobs = chain
lnprobability  = h5f['/emcee/lnprobability']
acceptance_fraction  = h5f['/emcee/acceptance_fraction']
acor                 = h5f['/emcee/acor']

print 'Acceptance Fraction for all walkers:'
print acceptance_fraction[:]

if kind_sin == 1: spline = h5f['/emcee/spline']

if flag_rv: rv_x0 = rv_x - Tref
if flag_p1: p1_x0 = p1_x - Tref
if flag_p2: p2_x0 = p2_x - Tref
if flag_fw: fw_x0 = fw_x - Tref
if flag_bs: bs_x0 = bs_x - Tref

## We start by plotting the chains
#out_med = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(sampler.flatchain[nburn:,:], #[16, 50, 84],axis=0)))
#for ii in xrange(0,ndim):
#  print out_med[ii]


### Start plotting the chains
print 'Tref = ',Tref


# chain is transposed
chain_T =  np.ndarray([nsteps,npop,ndim],dtype=np.double)
for ii in xrange(0,ndim):
  chain_T[:,:,ii] = chain[:,:,ii].T



lnprb_T =  lnprobability[:][:].T


chain_burnt = chain_T[nburn:,:,:]
blobs_burnt = blobs[nburn:,:,:]
lnprb_burnt = lnprb_T[nburn:,:]

#INSERIRE ELEMENTO DI THINNING!!!!

#flattening the blobs
s = chain_burnt.shape
flatchain = chain_burnt.reshape(s[1] * s[0], s[2])
s = blobs_burnt[:].shape
flatblobs = blobs_burnt.reshape(s[1] * s[0], s[2])
s = lnprb_burnt[:].shape
flatlnprb = lnprb_burnt.reshape(s[1] * s[0])

n_kept = s[1] * s[0]

#index of maximum probability values
maxlnprob_ind = flatlnprb.argmax()


#Obtaining the 1-sigma errors on values
chain_med = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
              zip(*np.percentile(flatchain[:,:], [15.865, 50, 84.135],axis=0)))

blobs_med = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
              zip(*np.percentile(flatblobs[:,:], [15.865, 50, 84.135],axis=0)))


ip = 0
id = 0

id = 0
ip = 0

orbit_pam = np.zeros((n_planets,5),dtype=np.double)
system_pam = np.zeros(n_rvj+n_rvo+n_rvl)

if flag_rv:
  for pp in xrange(0,n_planets):
    # Period
    fig=plt.plot(blobs[:,:,id], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_planet'+`pp`+'_derived_P_chain.png', bbox_inches='tight')
    plt.close()
    print 'Planet '+`pp`+' Period = ',blobs_med[id],' Acor= ',acor[ip]
    orbit_pam[pp,0] = blobs_med[id][0]
    id += 1
    ip += 1

    # Semi-Amplitude
    fig=plt.plot(blobs[:,:,id], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_planet'+`pp`+'_derived_K_chain.png', bbox_inches='tight')
    plt.close()
    print 'Planet '+`pp`+' K      = ',blobs_med[id],' Acor= ',acor[ip]
    orbit_pam[pp,1] = blobs_med[id][0]

    id += 1
    ip += 1

    # Phase
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_planet'+`pp`+'_phase_chain.png', bbox_inches='tight')
    plt.close()
    print 'Planet '+`pp`+' Phase  = ',np.asarray(chain_med[ip]),' Acor= ',acor[ip]
    orbit_pam[pp,2] = chain_med[ip][0]
    ip += 1

    if (n_orbpams[pp] == 5):
      # eccentricity
      fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
      plt.savefig('output/'+planet_name+'_planet'+`pp`+'_sqe_sinomega_chain.png', bbox_inches='tight')
      plt.close()
      print 'Planet '+`pp`+' sqe_sinomega = ',chain_med[ip],' Acor= ',acor[ip]
      ip += 1

      fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
      plt.savefig('output/'+planet_name+'_planet'+`pp`+'_sqe_cosomega_chain.png', bbox_inches='tight')
      plt.close()
      print 'Planet '+`pp`+' sqe_cosomega = ',chain_med[ip],' Acor= ',acor[ip]
      ip += 1

      fig=plt.plot(blobs[:,:,id], '-', color='k', alpha=0.3)
      plt.savefig('output/'+planet_name+'_planet'+`pp`+'_derived_e_chain.png', bbox_inches='tight')
      plt.close()
      print 'Planet '+`pp`+' e      = ',blobs_med[id]
      orbit_pam[pp,3] = blobs_med[id][0]
      id += 1

      # Omega
      fig=plt.plot(blobs[:,:,id], '-', color='k', alpha=0.3)
      plt.savefig('output/'+planet_name+'_planet'+`pp`+'_derived_omega_chain.png', bbox_inches='tight')
      plt.close()
      print 'Planet '+`pp`+' omega  = ',blobs_med[id]
      orbit_pam[pp,4] = blobs_med[id][0]
      id += 1

  for ii in xrange(0,n_rvj):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_rv_jitter_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic RV jitter '+`ii`+' = ',chain_med[ip], ' Acor= ',acor[ip]
    system_pam[ii] = chain_med[ip][0]
    ip += 1

  for ii in xrange(0,n_rvo):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_rv_offset_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic RV offset '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    system_pam[n_rvj+ii] = chain_med[ip][0]
    ip += 1

  for ii in xrange(0,n_rvl):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_rv_trend_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic RV trend '+`ii`+' = ',chain_med[ip],\
       ' Acor= ',acor[ip]
    system_pam[n_rvj+n_rvo+ii] = chain_med[ip][0]
    ip += 1


if flag_p1:
  for ii in xrange(0,n_p1j):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_p1_jitter_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic P1 jitter '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

  for ii in xrange(0,n_p1o):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_p1_offset_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic P1 offset '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

  for ii in xrange(0,n_p1l):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_p1_trend_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic P1 trend '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

if flag_p2:
  for ii in xrange(0,n_p2j):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_p2_jitter_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic P2 jitter '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

  for ii in xrange(0,n_p2o):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_p2_offset_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic P2 offset '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

  for ii in xrange(0,n_p2l):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_p2_trend_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic P2 trend '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

if flag_fw:
  for ii in xrange(0,n_fwj):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_fw_jitter_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic FW jitter '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

  for ii in xrange(0,n_fwo):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_fw_offset_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic FW offset '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

  for ii in xrange(0,n_fwl):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_fw_trend_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic FW trend '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

if flag_bs:
  for ii in xrange(0,n_bsj):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_bs_jitter_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic BS jitter '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

  for ii in xrange(0,n_bso):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_bs_offset_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic BS offset '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

  for ii in xrange(0,n_bsl):
    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_bs_trend_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print 'Systemic BS trend '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

## external parameters
if (kind_sin > 0):



    fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
    plt.savefig('output/'+planet_name+'_Prot_'+`ii`+'.png', bbox_inches='tight')
    plt.close()
    print '  SinFit Period = ',chain_med[ip],' Acor= ',acor[ip]
    ip += 1

    for ii in xrange(0,n_pof):
        fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
        plt.savefig('output/'+planet_name+'_pof_'+`ii`+'.png', bbox_inches='tight')
        plt.close()
        print '  SinFit Offsets '+`ii`+' = ',np.asarray(chain_med[ip])*360.0,' Acor= ',acor[ip]
        ip += 1

    for nk in xrange(0, n_periods):

      for ii in xrange(0,n_pha):
        fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
        plt.savefig('output/'+planet_name+'_Period_'+`nk`+'_pha_'+`ii`+'.png', bbox_inches='tight')
        plt.close()
        print 'Period ',nk, '    SinFit Phases '+`ii`+' = ',np.asarray(chain_med[ip])*360.0,' Acor= ',acor[ip]
        ip += 1

      if rvp_flag[nk]:
        for jj in xrange(0, n_rva):
          for ii in xrange(0,n_amp[jj]):
            fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
            plt.savefig('output/'+planet_name+'_Period_'+`nk`+'_SinFit_'+`jj`+'_ampRV_'+`ii`+'.png', bbox_inches='tight')
            plt.close()
            print 'Period ',nk, '    SinFit curve '+`jj`+' ampRV '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
            ip += 1

      if p1p_flag[nk]:
        for ii in xrange(0,n_pho):
          fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
          plt.savefig('output/'+planet_name+'_Period_'+`nk`+'_ampP1_'+`ii`+'.png', bbox_inches='tight')
          plt.close()
          print 'Period ',nk, '    SinFit ampP1 '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
          ip += 1

      if p2p_flag[nk]:
        for ii in xrange(0,n_pho):
          fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
          plt.savefig('output/'+planet_name+'_Period_'+`nk`+'_ampP2_'+`ii`+'.png', bbox_inches='tight')
          plt.close()
          print 'Period ',nk, '    SinFit ampP2 '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
          ip += 1

      if fwp_flag[nk]:
        for jj in xrange(0, n_fwa):
          for ii in xrange(0,n_amp[jj]):
            fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
            plt.savefig('output/'+planet_name+'_Period_'+`nk`+'_SinFit_'+`jj`+'_ampFW_'+`ii`+'.png', bbox_inches='tight')
            plt.close()
            print 'Period ',nk, '    SinFit curve '+`jj`+' ampFW '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
            ip += 1

      if bsp_flag[nk]:
        for jj in xrange(0,n_bsa):
          for ii in xrange(0,n_amp[jj]):
            fig=plt.plot(chain_T[:,:,ip], '-', color='k', alpha=0.3)
            plt.savefig('output/'+planet_name+'_Period_'+`nk`+'_SinFit_'+`jj`+'_ampBS_'+`ii`+'.png', bbox_inches='tight')
            plt.close()
            print 'Period ',nk, '    SinFit curve '+`jj`+' ampBS '+`ii`+' = ',chain_med[ip],' Acor= ',acor[ip]
            ip += 1

## RVs plot
labels_sys = []
labels_all = []
labels_plt = []

sample_sys = np.zeros([n_kept,n_rvj+n_rvo])
ip = np.sum(n_orbpams, dtype=np.int64)
ic = 0
for ii in xrange(0,n_rvj):
  labels_sys += ["RV$_{jit}$"+`ii`]
  sample_sys[:,ii] = flatchain[:,ip+ii]
ic += n_rvj
ip += n_rvj
for ii in xrange(0,n_rvo):
  labels_sys += ["$\gamma$"+`ii`]
  sample_sys[:,ic+ii] = flatchain[:,ip+ii]

labels_orb5 = ["P", "K", "Phase", "e", "$\omega$"]
labels_orb3 = ["P", "K", "Phase"]

#Triangle plot for each planet
id = 0
ip = 0
for pp in xrange(0,n_planets):
  labels_plt = []
  plot_truths = np.zeros(n_orbpams[pp]+n_rvj+n_rvo+n_rvl)

  sample_plan = np.zeros([n_kept,n_orbpams[pp]+n_rvj+n_rvo+n_rvl])
  sample_plan[:,0] = flatblobs[:,id+0]
  sample_plan[:,1] = flatblobs[:,id+1]
  sample_plan[:,2] = flatchain[:,ip+2]

  plot_truths[0:3]=orbit_pam[pp,0:3]

  id += 2
  if (n_orbpams[pp]==5) :
    sample_plan[:,3] = flatblobs[:,id+0]
    sample_plan[:,4] = flatblobs[:,id+1]
    plot_truths[3] = orbit_pam[pp,3]
    plot_truths[4] = orbit_pam[pp,4]

    id += 2
    labels_plt +=  labels_orb5
  else:
    labels_plt +=  labels_orb3
  labels_plt += labels_sys
  sample_plan[:,n_orbpams[pp]:n_orbpams[pp]+(n_rvj+n_rvo+n_rvl)]=sample_sys
  plot_truths[n_orbpams[pp]:n_orbpams[pp]+(n_rvj+n_rvo+n_rvl)]=system_pam[:]
  ip += n_orbpams[pp]

  fig = triangle.corner(sample_plan, labels=labels_plt, truths=plot_truths)
  fig.savefig("output/"+planet_name+"_planet"+`pp`+"_RVtriangle_wp_wt.png", bbox_inches='tight')
  plt.close()
  #
  fig = triangle.corner(sample_plan, labels=labels_plt, truths=plot_truths, plot_contours=True, plot_datapoints=False)
  fig.savefig("output/"+planet_name+"_planet"+`pp`+"_RVtriangle.png", bbox_inches='tight')
  plt.close()


#Plots for kind_sin = 1 selection

# if (kind_sin>0): labels_sys += ["$\Prot$$"+`ii`]



#

#OC101_fast.config
