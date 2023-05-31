import numpy as np

from cosmosis.datablock import option_section

import abundance_likelihood

def setup(options):
    M_lim = options.get_double_array_1d(option_section, 'M_lim')
    z_lim = options.get_double_array_1d(option_section, 'z_lim')
    lnM_err = options.get_double(option_section, 'lnM_err', default=0.)
    survey_area = options.get_double(option_section, 'survey_area')
    fname = options.get_string(option_section, 'dataframe')
    with open(fname, 'r') as f:
        header_names = f.readline().split()[1:]
    dataframe = np.genfromtxt(fname, dtype=None, names=header_names)
    return {'dataframe': dataframe, 'M_lim': M_lim, 'z_lim': z_lim, 'survey_area': survey_area, 'lnM_err': lnM_err}

def execute(block, setup_stuff):
    z, M, N = block.get_grid('HMF', 'z_arr', 'M_arr', 'dNdlnM')
    HMF = {'z_arr': z, 'M_arr': M, 'dNdlnM': N}

    cosmology = {
                 'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
                 'Omega_b': block.get_double('cosmological_parameters', 'Omega_b'),
                 'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
                 'w0': block.get_double('cosmological_parameters', 'w'),
                 'wa': block.get_double('cosmological_parameters', 'wa'),
                 'h': block.get_double('cosmological_parameters', 'hubble')/100,
                 'sigma8': block.get_double('cosmological_parameters', 'sigma_8'),
                 'ns': block.get_double('cosmological_parameters', 'n_s'),
                 'mass_bias': block.get_double('cosmological_parameters', 'mass_bias'),
    }

    lnlike = abundance_likelihood.lnlike(cosmology, setup_stuff['M_lim'], setup_stuff['z_lim'], HMF, setup_stuff['dataframe'], setup_stuff['survey_area'], setup_stuff['lnM_err'])
    # print('lnlike', lnlike)
    block.put_double('likelihoods', 'abundance_LIKE', lnlike)

    return 0

def cleanup(config):
    pass
