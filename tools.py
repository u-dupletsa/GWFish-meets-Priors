#    Copyright (c) 2024 Ulyana Dupletsa <ulyana.dupletsa@gssi.it>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.


import numpy as np
import pandas as pd
import h5py
import os
import pesummary
from pesummary.io import read
import pickle
import yaml
import GWFish.modules as gw
import json
from tqdm import tqdm
from minimax_tilting_sampler import *
import priors



def keys(f):
    """
    Get the keys of a hdf5 file
    """
    return [key for key in f.keys()]



def get_events_list(PATH_TO_DATA, PATH_TO_RESULTS, waveform = 'IMRPhenomXPHM'):
    """
    Get the list of events in the data directory
    satisfying the following criteria:
    - The event has the waveform IMRPhenomXPHM
    - The event has the estimator (median and/or maxP)
    - The event has the analytic priors
    """
    all_events = []
    events_without_wf = []
    events_without_median = []
    events_without_maxP = []
    events_without_priors = []
    events_wf_median_priors = []
    events_wf_maxP_priors = []
    events_wf_median_maxP_priors = []

    for file in os.listdir(PATH_TO_DATA):
        data_pesum = read(PATH_TO_DATA + file, package = 'core')
        data = h5py.File(PATH_TO_DATA + file, 'r') 
        if 'C01:' + waveform not in data_pesum.samples_dict.keys():
            events_without_wf.append(file[:-3])
        else:
            # try with one of the parameters to see if the estimator is None
            if data_pesum.samples_dict['C01:' + waveform].key_data['mass_1']['median'] == None:
                events_without_median.append(file[:-3])
            if data_pesum.samples_dict['C01:' + waveform].key_data['mass_1']['maxP'] == None:
                events_without_maxP.append(file[:-3])
            if 'analytic' not in data['C01:' + waveform]['priors'].keys(): 
                events_without_priors.append(file[:-3])
            if data_pesum.samples_dict['C01:' + waveform].key_data['mass_1']['median'] != None\
               and 'analytic' in data['C01:' + waveform]['priors'].keys():
                events_wf_median_priors.append(file[:-3])
            if data_pesum.samples_dict['C01:' + waveform].key_data['mass_1']['maxP'] != None\
                and 'analytic' in data['C01:' + waveform]['priors'].keys():
                 events_wf_maxP_priors.append(file[:-3])
            if data_pesum.samples_dict['C01:' + waveform].key_data['mass_1']['median'] != None\
                and data_pesum.samples_dict['C01:' + waveform].key_data['mass_1']['maxP'] != None\
                and 'analytic' in data['C01:' + waveform]['priors'].keys():
                events_wf_median_maxP_priors.append(file[:-3])

    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_without_wf_%s.txt' %waveform, events_without_wf, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_without_median_%s.txt' %waveform, events_without_median, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_without_maxP_%s.txt' %waveform, events_without_maxP, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_without_priors_%s.txt' %waveform, events_without_priors, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_wf_median_priors_%s.txt' %waveform, events_wf_median_priors, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_wf_maxP_priors_%s.txt' %waveform, events_wf_maxP_priors, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_wf_median_maxP_priors_%s.txt' %waveform, events_wf_median_maxP_priors, fmt = '%s')

            

def get_injections_from_gwtc(PATH_TO_DATA, PATH_TO_INJECTIONS, events_list, waveform, params, estimator):
    """
    Load the GWTC data and extract the injections
    The injections are saved in a hdf5 file
        --> Specify the PATH_TO_DATA and PATH_TO_RESULTS
        --> Specify the waveform, the parameters and the estimator
    The DATA are assumed to be in the LVK format as can be downloaded from Zenodo
    """

    for event in events_list:
        data = read(PATH_TO_DATA + event + '.h5', package = 'core')
        # Create the injections
        estimator_dict = {}
        for param in params:
            estimator_dict[param] = data.samples_dict['C01:' + waveform].key_data[param][estimator]
        if not os.path.exists(PATH_TO_INJECTIONS + event):
            os.makedirs(PATH_TO_INJECTIONS + event)
        estimator_df = pd.DataFrame([estimator_dict], columns = params)
        estimator_df.to_hdf(PATH_TO_INJECTIONS + event + '/%s_%s_%s.hdf5' %(event, waveform, estimator), key = 'data', mode = 'w')



def check_and_store_chirp_mass_priors(PATH_TO_DATA, PATH_TO_RESULTS, events_list, waveform):
    """
    Check the chirp mass priors and store them in a dictionary
    """
    chirp_mass_priors = {}
    for event in events_list:
        data = h5py.File(PATH_TO_DATA + event + '.h5', 'r') 
        string_ov = data['C01:' + waveform]['priors']['analytic']['chirp_mass'][0].decode('utf-8')
        new_string = string_ov.replace('=', ',').split(',')
        min_chirp_mass = new_string[1]
        max_chirp_mass = new_string[3]
        chirp_mass_priors[event] = [min_chirp_mass, max_chirp_mass]

    with open(PATH_TO_RESULTS + 'info/' + 'chirp_mass_priors_%s.pkl' %waveform, 'wb') as f:
        pickle.dump(chirp_mass_priors, f)

def check_and_store_geocent_time_priors(PATH_TO_LVK_DATA, PATH_TO_RESULTS, events, waveform):
    """
    Check the geocent time priors and store them in a dictionary
    """
    geocent_time_priors = {}
    for event in events:
        data = h5py.File(PATH_TO_LVK_DATA + event + '.h5', 'r')
        string_ov = data['C01:' + waveform]['priors']['analytic']['geocent_time'][0].decode('utf-8')
        new_string = string_ov.replace('=', ',').split(',')
        min_geocent_time = new_string[1]
        max_geocent_time = new_string[3]
        geocent_time_priors[event] = [min_geocent_time, max_geocent_time]

    with open(PATH_TO_RESULTS + 'info/' + 'geocent_time_priors_%s.pkl' %waveform, 'wb') as f:
        pickle.dump(geocent_time_priors, f)

def detectors_and_yaml_files(PATH_TO_DATA, PATH_TO_RESULTS, PATH_TO_YAML, PATH_TO_PSD, events_list, waveform):
    """
    Store information about detectors in .yaml files
    """
    # Create generic dictionary template in the format taken in GWFish
    dict_template = {'L1':{'lat':30.56 * np.pi / 180.,
                    'lon':-90.77 * np.pi / 180.,
                    'opening_angle':np.pi / 2.,
                    'azimuth':197.7 * np.pi / 180.,
                    'duty_factor':0.85,
                    'detector_class':'earthL',
                    'plotrange':'10, 1000, 1e-25, 1e-20',
                    'fmin':2,
                    'fmax':2048,
                    'spacing':'geometric',
                    'df':1/4,
                    'npoints':5000,
                    'arm_length': 4000
                    },
               'H1':{'lat':46.45 * np.pi / 180.,
                    'lon':-119.41 * np.pi / 180.,
                    'opening_angle':np.pi / 2.,
                    'azimuth':171.8 * np.pi / 180.,
                    'duty_factor':0.85,
                    'detector_class':'earthL',
                    'plotrange':'10, 1000, 1e-25, 1e-20',
                    'fmin':2,
                    'fmax':2048,
                    'spacing':'geometric',
                    'df':1/4,
                    'npoints':5000,
                    'arm_length': 4000
                    },
                'V1':{'lat':43.63 * np.pi / 180.,
                    'lon':10.51 * np.pi / 180.,
                    'opening_angle':np.pi / 2.,
                    'azimuth':116.5 * np.pi / 180.,
                    'duty_factor':0.85,
                    'detector_class':'earthL',
                    'plotrange':'10, 1000, 1e-25, 1e-20',
                    'fmin':2,
                    'fmax':2048,
                    'spacing':'geometric',
                    'df':1/4,
                    'npoints':5000,
                    'arm_length': 3000
                    }
            }

    detectors = {}
    for event in events_list:

        local_dictionary = dict_template.copy()

        data = h5py.File(PATH_TO_DATA + event + '.h5', 'r')
        detectors[event] = keys(data['C01:' + waveform]['psds'])

        for j in range(len(detectors[event])):
            local_dictionary[detectors[event][j]].update({'psd_data':PATH_TO_PSD + 'psd_%s_%s_%s.txt' %(waveform, event, detectors[event][j])})

            np.savetxt(PATH_TO_PSD + 'psd_%s_%s_%s.txt' %(waveform, event, detectors[event][j]), 
                       np.c_[data['C01:' + waveform]['psds'][detectors[event][j]][:, 0], 
                             data['C01:' + waveform]['psds'][detectors[event][j]][:, 1]])
                                                               
        with open(PATH_TO_YAML + '%s.yaml' %event, 'w') as my_yaml_file:
            yaml.dump(local_dictionary, my_yaml_file)

    
    with open(PATH_TO_RESULTS + 'info/' + 'detectors_dictionary.pkl', 'wb') as f:
        pickle.dump(detectors, f)



def get_label(detectors_list, event, estimator, snr_thr, name_tag, additional_tag = None):
    """
    Provide fast way to create labels and retrieve file names
    """
    detectors_labels = list(detectors_list[event])
    connector = '_'
    network_lbs = detectors_labels[0]
    for j in range(1, len(detectors_labels)):
        network_lbs += connector + detectors_labels[j]
    if estimator == 'averaged':
        if name_tag == 'errors':
            if additional_tag == None:
                label = 'Errors_%s_BBH_%s_SNR%s.txt' %(network_lbs, event, snr_thr)
            else:
                label = 'Errors_%s_%s_BBH_%s_SNR%s.txt' %(network_lbs, additional_tag, event, snr_thr)
        elif name_tag == 'fishers':
            if additional_tag == None:
                label = 'fisher_matrices_%s_BBH_%s_SNR%s.npy' %(network_lbs, event, snr_thr)
            else:
                label = 'fisher_matrices_%s_%s_BBH_%s_SNR%s.npy' %(network_lbs, additional_tag, event, snr_thr)
        elif name_tag == 'inv_fishers':
            if additional_tag == None:
                label = 'inv_fisher_matrices_%s_BBH_%s_SNR%s.npy' %(network_lbs, event, snr_thr)
            else:
                label = 'inv_fisher_matrices_%s_%s_BBH_%s_SNR%s.npy' %(network_lbs, additional_tag, event, snr_thr)

    else:
        if name_tag == 'errors':
            if additional_tag == None:
                label = 'Errors_%s_%s_BBH_%s_SNR%s.txt' %(network_lbs, estimator, event, snr_thr)
            else:
                label = 'Errors_%s_%s_%s_BBH_%s_SNR%s.txt' %(network_lbs, estimator, additional_tag, event, snr_thr)
        elif name_tag == 'fishers':
            if additional_tag == None:
                label = 'fisher_matrices_%s_%s_BBH_%s_SNR%s.npy' %(network_lbs, estimator, event, snr_thr)
            else:
                label = 'fisher_matrices_%s_%s_%s_BBH_%s_SNR%s.npy' %(network_lbs, estimator, additional_tag, event, snr_thr)
        elif name_tag == 'inv_fishers':
            if additional_tag == None:
                label = 'inv_fisher_matrices_%s_%s_BBH_%s_SNR%s.npy' %(network_lbs, estimator, event, snr_thr)
            else:
                label = 'inv_fisher_matrices_%s_%s_%s_BBH_%s_SNR%s.npy' %(network_lbs, estimator, additional_tag, event, snr_thr)
    return label



def gwfish_analysis(PATH_TO_INJECTIONS, PATH_TO_YAML, PATH_TO_RESULTS, events_list, waveform, estimator,
                    detectors, fisher_parameters):
    """
    Perform the Fisher matrix analysis using GWFish
    """

    for event in events_list:
        name_tag = '%s_BBH_%s' %(estimator, event)

        detectors_list = detectors[event]
        detectors_event = []
        for j in range(len(detectors_list)):
            detectors_event.append(detectors_list[j])
        networks = np.linspace(0, len(detectors_event) - 1, len(detectors_event), dtype=int)
        list_all_networks = []
        for i in range(len(networks)):
            list_all_networks.append([networks[i].tolist()])
        list_all_networks.append(networks.tolist()) 
        networks = str(list_all_networks)

        detectors_ids = np.array(detectors_event)
        networks_ids = json.loads(networks)
        ConfigDet = os.path.join(PATH_TO_YAML + event + '.yaml')

        gw_parameters = pd.read_hdf(PATH_TO_INJECTIONS + event +  '/%s_%s_%s.hdf5' %(event, waveform, estimator))

        network = gw.detection.Network(detectors_ids, detection_SNR=(0., 0.), config=ConfigDet)
        gw.fishermatrix.analyze_and_save_to_txt(network = network,
                                        parameter_values  = gw_parameters,
                                        fisher_parameters = fisher_parameters, 
                                        sub_network_ids_list = networks_ids,
                                        population_name = name_tag,
                                        waveform_model = waveform,
                                        save_path = PATH_TO_RESULTS + event +'/',
                                        save_matrices = True,
                                        decimal_output_format='%.15f')



def from_m1_m2_to_mChirp_q(m1, m2):
    """
    Compute the transformation from m1, m2 to mChirp, q
    """
    mChirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    q = m2 / m1
    return mChirp, q



def from_mChirp_q_to_m1_m2(mChirp, q):
    """
    Compute the transformation from mChirp, q to m1, m2
    """
    m1 = mChirp * (1 + q)**(1/5) * q**(-3/5)
    m2 = mChirp * (1 + q)**(1/5) * q**(2/5)
    return m1, m2



def derivative_m1_m2_dmChirp_dq(m1, m2, mChirp, q):
    """
    Compute the derivative of m1, m2 with respect to mChirp, q
    """
    dm1_dmChirp = (1 + q)**(1/5) * q**(-3/5)
    dm1_dq = mChirp * (1 + q)**(1/5) * (-3/5) * q**(-8/5) + mChirp * (1/5) * (1 + q)**(-4/5) * q**(-3/5)
    dm2_dmChirp = (1 + q)**(1/5) * q**(2/5)
    dm2_dq = mChirp * (1 + q)**(1/5) * (2/5) * q**(-3/5) + mChirp * (1/5) * (1 + q)**(-4/5) * q**(2/5)
    return dm1_dmChirp, dm1_dq, dm2_dmChirp, dm2_dq



def jacobian_for_derivative_from_m1_m2_to_mChirp_q(m1, m2, fisher_matrix):
    """
    Compute the Jacobian for the transformation from m1, m2 to mChirp, q
    """
    mChirp, q = from_m1_m2_to_mChirp_q(m1, m2)
    dm1_dmChirp, dm1_dq, dm2_dmChirp, dm2_dq = derivative_m1_m2_dmChirp_dq(m1, m2, mChirp, q)
    rotated_fisher = fisher_matrix.copy()
    jacobian_matrix = np.zeros_like(fisher_matrix)
    nparams = len(fisher_matrix[0, 0, :])
    for i in range(nparams):
        jacobian_matrix[0, i, i] = 1
    jacobian_matrix[0, 0, 0] = dm1_dmChirp
    jacobian_matrix[0, 0, 1] = dm1_dq
    jacobian_matrix[0, 1, 0] = dm2_dmChirp
    jacobian_matrix[0, 1, 1] = dm2_dq

    rotated_fisher = jacobian_matrix[0, :, :].T @ rotated_fisher[0, :, :] @ jacobian_matrix[0, :, :]

    return rotated_fisher[np.newaxis, :, :]



def get_rotated_fisher_matrix(PATH_TO_RESULTS, events_list, detectors_list, estimator, lbs_errs, new_fisher_parameters):
    """
    Compute the rotated Fisher matrix from m1, m2 to mChirp, q
    """
   
    for event in events_list:

        label = get_label(detectors_list, event, estimator, 'errors')

        signals = pd.read_csv(PATH_TO_RESULTS + 'results/gwfish_m1_m2/' +
                            get_label(detectors_list, event, estimator, 'errors'), names = lbs_errs, skiprows = 1,
                            delimiter = ' ')
        
        fishers = np.load(PATH_TO_RESULTS + 'results/gwfish_m1_m2/' + 
                         get_label(detectors_list, event, estimator, 'fishers'))
        m1, m2 = signals[['mass_1', 'mass_2']].iloc[0]
        rotated_fisher = jacobian_for_derivative_from_m1_m2_to_mChirp_q(m1, m2, fishers)
        np.save(PATH_TO_RESULTS + 'results/gwfish_rotated/' + 
                get_label(detectors_list, event, estimator, 'fishers'), rotated_fisher)

        inv_rotated_fisher, _ = gw.fishermatrix.invertSVD(rotated_fisher[0, :, :])
        np.save(PATH_TO_RESULTS + 'results/gwfish_rotated/' + 
                get_label(detectors_list, event, estimator, 'inv_fishers'), inv_rotated_fisher)
        
        new_errors = signals.copy()

        err_params = []
        for l in range(len(new_fisher_parameters)):
            err_params.append('err_' + new_fisher_parameters[l])
        new_errors[err_params] = np.sqrt(np.diag(inv_rotated_fisher))
        np.savetxt(PATH_TO_RESULTS + 'results/gwfish_rotated/' +
                get_label(detectors_list, event, estimator, 'errors'), new_errors, delimiter = ' ', 
                fmt = '%.15f', header = '# ' + ' '.join(new_errors.keys()), comments = '')



def get_samples_from_TMVN(min_array, max_array, means, cov, N):
    """
    Draw samples from a truncated multivariate normal distribution
    """
    tmvn = TruncatedMVN(means, cov, min_array, max_array)
    return tmvn.sample(N)



def get_posteriors(samples, priors_dict, N):
    """
    Draw samples from a multivariate normal distribution with priors
    """
    # account for the fact that the prior on masses should be uniform in m1-m2 plane
    samples['priors'] = priors.uniform_pdf(samples['chirp_mass'].to_numpy(), priors_dict['chirp_mass'][0], priors_dict['chirp_mass'][1])*\
                        priors.uniform_pdf(samples['mass_ratio'].to_numpy(), priors_dict['mass_ratio'][0], priors_dict['mass_ratio'][1])*\
                        (samples['chirp_mass'].to_numpy()*(samples['mass_ratio'])**(-6./5.)*(1+samples['mass_ratio'].to_numpy())**(2/5))*\
                        priors.uniform_in_distance_squared_pdf(samples['luminosity_distance'].to_numpy(), priors_dict['luminosity_distance'][0], priors_dict['luminosity_distance'][1])*\
                        priors.cosine_pdf(samples['dec'].to_numpy(), priors_dict['dec'][0], priors_dict['dec'][1])*\
                        priors.uniform_pdf(samples['ra'].to_numpy(), priors_dict['ra'][0], priors_dict['ra'][1])*\
                        priors.sine_pdf(samples['theta_jn'].to_numpy(), priors_dict['theta_jn'][0], priors_dict['theta_jn'][1])*\
                        priors.uniform_pdf(samples['psi'].to_numpy(), priors_dict['psi'][0], priors_dict['psi'][1])*\
                        priors.uniform_pdf(samples['phase'].to_numpy(), priors_dict['phase'][0], priors_dict['phase'][1])*\
                        priors.uniform_pdf(samples['geocent_time'].to_numpy(), priors_dict['geocent_time'][0], priors_dict['geocent_time'][1])*\
                        priors.uniform_pdf(samples['a_1'].to_numpy(), priors_dict['a_1'][0], priors_dict['a_1'][1])*\
                        priors.uniform_pdf(samples['a_2'].to_numpy(), priors_dict['a_2'][0], priors_dict['a_2'][1])*\
                        priors.sine_pdf(samples['tilt_1'].to_numpy(), priors_dict['tilt_1'][0], priors_dict['tilt_1'][1])*\
                        priors.sine_pdf(samples['tilt_2'].to_numpy(), priors_dict['tilt_2'][0], priors_dict['tilt_2'][1])*\
                        priors.uniform_pdf(samples['phi_12'].to_numpy(), priors_dict['phi_12'][0], priors_dict['phi_12'][1])*\
                        priors.uniform_pdf(samples['phi_jl'].to_numpy(), priors_dict['phi_jl'][0], priors_dict['phi_jl'][1])

    samples['weights'] = samples['priors'] / np.sum(samples['priors'])
    prob = samples['weights'].to_numpy()
    index = np.random.choice(np.arange(N), size = N, replace = True, p = prob)
    posteriors = samples.iloc[index]
    
    return posteriors



def get_lvk_samples(PATH_TO_LVK_DATA, event, params):
    """
    Get the LVK samples
    """
    data = h5py.File(PATH_TO_LVK_DATA + event + '.h5', 'r')
    samples_lvk = {}
    for l in range(len(params)):
        samples_lvk[params[l]] = data['C01:IMRPhenomXPHM']['posterior_samples'][params[l]]

    return samples_lvk



def get_confidence_interval(samples, params, confidence_level):
    """
    Compute the confidence intervals
    """
    confidence_level /= 100
    conf_int = {}
    for param in params:
        conf_int[param] = np.percentile(samples[param], [100 * (1 - confidence_level) / 2, 100 * (1 + confidence_level) / 2])
    
    return conf_int


def my_multivariate_normal(mean, cov, n_samples, epsilon = 1e-10):
    """
    Draw samples from a multivariate normal distribution
    """

    d = int(len(mean))
    perturbed_cov = cov + epsilon * np.identity(d)
    # Cholesky matrix
    L = np.linalg.cholesky(perturbed_cov)
    u = np.random.normal(loc = 0, scale = 1, size = d * n_samples).reshape(d, n_samples)

    return np.squeeze(mean + L.dot(u).T)
