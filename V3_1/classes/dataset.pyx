from common import *

class Dataset:
    def __init__(self, ind, kind, input_file, models):
        self.ind = ind
        self.kind = kind
        # 'RV', 'PHOT', 'ACT'...
        self.models = models
        self.name_ref = input_file

        self.list_sys = ['jitter', 'offset', 'linear']
        self.bounds = {}
        self.starts = {}
        self.n_sys = {}

        """There is a mix of arrays and dictonaries here because this was one of the first
            classes I created. I will probably consider switch everything to dictionary
            without affecting the functionality in a later time"""

        print 'Opening: ', input_file
        self.data = np.loadtxt(input_file)

        n_cols = np.size(self.data, axis=1)

        self.x = np.asarray(self.data[:, 0], dtype=np.double)
        self.y = np.asarray(self.data[:, 1], dtype=np.double)
        self.e = np.asarray(self.data[:, 2], dtype=np.double)

        self.n = np.size(self.x)

        self.sys = {}
        for var in self.list_sys:
            self.sys[var] = np.zeros(self.n, dtype=np.double) - 1
        # it was zero for jitter and offset, and -1 for linear for unknown reasons

        if n_cols > 3:
            self.sys['jitter'] = np.asarray(self.data[:, 3], dtype=np.double)
        if n_cols > 4:
            self.sys['offset'] = np.asarray(self.data[:, 4], dtype=np.double)
        if n_cols > 5:
            self.sys['linear'] = np.asarray(self.data[:, 5], dtype=np.double)

        # use different offsets for the data
        # off must start from zero
        # -1 values for self.j and self.l mean that these will not be used

        # self.a = np.asarray(self.data[:, 5], dtype=np.double)
        # Flag for activity fit: we can choose to fit the same RV amplitude
        # of the activity signal for all the datasets, use dfferent values
        # for each dataset or exclude some of the datasets

        # Model for RV systematics
        for var in self.list_sys:
            self.n_sys[var] = np.max(self.sys[var].astype(np.int64)) + 1

        print 'N = ', self.n
        for var in self.list_sys:
            print 'N '+var+' = ', self.n_sys[var]
        # print 'N activ. = ', self.n_a
        print

        self.Tref = np.mean(self.x, dtype=np.double)
        self.x0 = self.x - self.Tref

        self.mask = {}
        for var in self.list_sys:
            self.mask[var] = np.zeros([self.n, self.n_sys[var]], dtype=bool)
            for ii in xrange(0, self.n_sys[var]):
                self.mask[var][(abs(self.sys[var] - ii) < 0.1), ii] = True

        self.model = np.zeros(self.n, dtype=np.double)
        self.jitter = np.zeros(self.n, dtype=np.double)

        """Default boundaries are defined according to the characteristic of the dataset"""
        self.default_bounds = {'offset': [np.min(self.y)-50., np.max(self.y)+50.],
                               'jitter': [0., 50 * np.max(self.e)],
                               'linear': [-1., 1.]}

    def common_Tref(self, Tref_in):
        self.Tref = Tref_in
        self.x0 = self.x - self.Tref
        return

    def model_reset(self):
        self.model[:] = 0.0
        self.jitter[:] = 0.0
        return

    def model_offset(self, off_in):
        off = np.atleast_1d(off_in)
        for ii in xrange(0, self.n_sys['offset']):
            self.model[self.mask['offset'][:, ii]] += off[ii]

    def model_linear(self, m_in):
        m = np.atleast_1d(m_in)
        for ii in xrange(0, self.n_sys['linear']):
            self.model[self.mask['linear'][:, ii]] += m[ii] * self.x0[self.mask['linear'][:, ii]]

    def model_jitter(self, jit_in):
        jit = np.atleast_1d(jit_in)
        for ii in xrange(0, self.n_sys['jitter']):
            self.jitter[self.mask['jitter'][:, ii]] = jit[ii]

    def model_logchi2(self):
        env = 1.0 / (self.e ** 2.0 + self.jitter ** 2.0)
        return -0.5 * (np.sum((self.y - self.model) ** 2 * env - np.log(env)))

    def define_bounds(self, mc):

        for var in self.list_sys:
            if var in self.bounds:
                bounds = self.bounds[var]
            else:
                bounds = self.default_bounds[var]
            for jj in xrange(0, self.n_sys[var]):
                mc.bounds_list.append(bounds)  # bounds for jitter

        mc.variable_list[self.kind] = {}
        mc.variable_list[self.name_ref] = {}

        """adding the systematics variables to the list"""
        for var in self.list_sys:
            mc.variable_list[self.name_ref][var] = np.arange(mc.ndim, mc.ndim + self.n_sys[var], 1)
            mc.ndim += self.n_sys[var]

    def starting_point(self, mc):
        if self.name_ref in self.starts:
            for var in self.starts[self.name_ref]:
                mc.starting_point[mc.variable_list[self.name_ref][var]] = self.starts[self.name_ref][var]
        """Better start with a low jitter and let the code increase it"""
        if 'jitter' in mc.variable_list[self.name_ref]:
            mc.starting_point[mc.variable_list[self.name_ref]['jitter']] = 0.0001

    def print_vars(self, mc, theta):
        for param in self.list_sys:
            id_var = mc.variable_list[self.name_ref][param]
            if np.size(id_var) == 0:
                continue
            if np.size(id_var) == 1:
                mc.pam_names[id_var[0]] = self.name_ref[:-4] + '_' + param
            else:
                for ii in id_var:
                    mc.pam_names[ii] = self.name_ref[:-4] + '_' + param + '_' + repr(ii - id_var[0])

            print self.name_ref, param, ' : ', theta[mc.variable_list[self.name_ref][param]]


class TransitCentralTimes(Dataset):
    def __init__(self, planet_name, input_file):

        self.kind = 'Tcent'
        self.models = ['Tcent']
        # 'RV', 'PHOT', 'ACT'...
        self.name_ref = input_file
        self.planet_name = planet_name

        self.list_sys = ['jitter', 'offset', 'linear']
        self.n_sys = {}
        self.mask = {}
        self.bounds = {}
        self.starts = {}
        self.deltaT = 1.10

        print 'Opening: ', input_file
        self.data = np.atleast_2d(np.loadtxt(input_file))

        self.n_transit = np.asarray(self.data[:, 0], dtype=np.int16)
        self.x = np.asarray(self.data[:, 1], dtype=np.double)
        self.e = np.asarray(self.data[:, 2], dtype=np.double)

        if np.size(self.data[0, :]) > 3:
            print 'TTV jitter found in the dataset'
            self.e = np.sqrt(self.data[:, 2] ** 2 + self.data[:, 3], dtype=np.double)

        self.n = np.size(self.x)
        self.model = np.zeros(self.n, dtype=np.double)

        self.Tref = np.mean(self.x, dtype=np.double)
        self.x0 = self.x - self.Tref

        print 'N = ', self.n
        print

        for var in self.list_sys:
            self.n_sys[var] = 0
            self.mask[var] = np.zeros([self.n, self.n_sys[var]], dtype=bool)

        self.model = np.zeros(self.n, dtype=np.double)
        self.jitter = np.zeros(self.n, dtype=np.double)

        """Default boundaries are defined according to the characteristic of the dataset"""
        self.default_bounds = {'offset': [np.min(self.x), np.max(self.x)],
                               'jitter': [0., 50 * np.max(self.e)],
                               'linear': [-1., 1.]}

    def compute(self, mc, theta):
        # By default, dataset.planet_name == planet_name
        dict_out = mc.pcv.convert(self.planet_name, theta)
        model = np.rint(self.x0 / dict_out['P'] - 1) * dict_out['P'] + \
                kp.kepler_Tcent_T0P(dict_out['P'], dict_out['f'], dict_out['e'], dict_out['o'])
        return model

    def model_logchi2(self):
        # boundaries in Tcent are specific of the dataset and not of a common
        # parameter for different dataset. The check can be internal
        # if np.sum(np.abs(self.x0 - self.model) < self.deltaT) < self.n:
        #    return -np.inf
        env = 1.0 / (self.e ** 2.0)
        time_dif = np.abs(self.x0 - self.model)
        # time_dif[np.where(time_dif > 6*self.e)] = 6*self.e
        # time_dif[np.where(time_dif > 8*self.e)] = 8*self.e + np.sqrt(np.abs(time_dif-8*self.e))
        return -0.5 * (np.sum(time_dif**2 * env - np.log(env)))

    def print_vars(self, mc, theta):
        # period, _, f, e, o = mc.pcv.convert(self.planet_name, theta)
        # model = np.rint(self.x0 / period) * period + kp.kepler_Tcent_T0P(period, f, e, o)

        if self.planet_name in mc.pcv.dynamical:
            dyn_output = mc.pcv.compute_dynamical(mc, theta)
            model = dyn_output[self.name_ref]
        else:
            model = self.compute(mc, theta)

        print 'Tc ', self.planet_name
        for ii in xrange(0, self.n):
            print 'Input Tc: ', self.x0[ii], '  Model Tc: ', model[ii], \
                '  Diff: ', model[ii] - self.x0[ii]
