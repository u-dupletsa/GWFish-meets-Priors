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



def keys(f):
    return [key for key in f.keys()]


def create_injections_from_gwtc(PATH_TO_DATA, PATH_TO_RESULTS, waveform, params, estimator):
    """
    Load the GWTC catalogs and create a list of events for which the estimator is not None
    The list of events is saved in a txt file (as well as the discareded ones)
    The injections are saved in a hdf5 file
        --> Specify the PATH_TO_DATA and PATH_TO_RESULTS
        --> Specify the waveform, the parameters and the estimator
    The DATA are assumed to be in the LVK format as can be downloaded from Zenodo
    """

    event_list = []
    no_waveform_list = []
    discarded_events_list = []
    for file in os.listdir(PATH_TO_DATA):
        data_pesum = read(PATH_TO_DATA + file, package = 'core')
        if 'C01:' + waveform not in data_pesum.samples_dict.keys():
            no_waveform_list.append(file[:-3])
        else:
            if data_pesum.samples_dict['C01:' + waveform].key_data[params[0]][estimator] != None:
                event_list.append(file[:-3])
                # Create the injections
                estimator_dict = {}
                for param in params:
                    estimator_dict[param] = data_pesum.samples_dict['C01:' + waveform].key_data[param][estimator]
                PATH_TO_INJECTIONS = PATH_TO_RESULTS + 'injections/' + file[:-3]
                if not os.path.exists(PATH_TO_INJECTIONS):
                    os.makedirs(PATH_TO_INJECTIONS)
                estimator_df = pd.DataFrame([estimator_dict], columns = params)
                estimator_df.to_hdf(PATH_TO_INJECTIONS + '/%s_%s_%s.hdf5' %(file[:-3], waveform, estimator), key = 'data', mode = 'w')
            else:
                discarded_events_list.append(file[:-3])

    np.savetxt(PATH_TO_RESULTS + 'info/' + 'event_list_%s_%s.txt' %(waveform, estimator), event_list, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + '%s_not_in_list_for_%s.txt' %(estimator, waveform), discarded_events_list, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + 'waveform_not_in_list_%s.txt' %waveform, no_waveform_list, fmt = '%s')


def check_and_store_priors(PATH_TO_DATA, PATH_TO_RESULTS, events_list, waveform):
    events_with_priors = []
    events_with_no_priors = []
    chirp_mass_priors = {}
    for event in events_list:
        data = h5py.File(PATH_TO_DATA + event + '.h5', 'r') 
        if 'analytic' in data['C01:' + waveform]['priors'].keys():
            events_with_priors.append(event)
            string_ov = data['C01:' + waveform]['priors']['analytic']['chirp_mass'][0].decode('utf-8')
            new_string = string_ov.replace('=', ',').split(',')
            min_chirp_mass = new_string[1]
            max_chirp_mass = new_string[3]
            chirp_mass_priors[event] = [min_chirp_mass, max_chirp_mass]
        else:
            events_with_no_priors.append(event)

    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_with_priors_%s.txt' %waveform, events_with_priors, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_with_no_priors_%s.txt' %waveform, events_with_no_priors, fmt = '%s')
    with open(PATH_TO_RESULTS + 'info/' + 'chirp_mass_priors_%s.pkl' %waveform, 'wb') as f:
        pickle.dump(chirp_mass_priors, f)

def detectors_and_yaml_files(PATH_TO_DATA, PATH_TO_RESULTS, PATH_TO_YAML, PATH_TO_PSD, events_list, waveform):

    dict_template = {'L1':{'lat':30.56 * np.pi / 180.,
                    'lon':-90.77 * np.pi / 180.,
                    'opening_angle':np.pi / 2.,
                    'azimuth':197.7 * np.pi / 180.,
                    'duty_factor':0.85,
                    'detector_class':'earthL',
                    'plotrange':'10, 1000, 1e-25, 1e-20',
                    'fmin':8,
                    'fmax':1024,
                    'spacing':'geometric',
                    'df':1/4,
                    'npoints':5000
                    },
               'H1':{'lat':46.45 * np.pi / 180.,
                    'lon':-119.41 * np.pi / 180.,
                    'opening_angle':np.pi / 2.,
                    'azimuth':171.8 * np.pi / 180.,
                    'duty_factor':0.85,
                    'detector_class':'earthL',
                    'plotrange':'10, 1000, 1e-25, 1e-20',
                    'fmin':8,
                    'fmax':1024,
                    'spacing':'geometric',
                    'df':1/4,
                    'npoints':5000
                    },
                'V1':{'lat':43.63 * np.pi / 180.,
                    'lon':10.51 * np.pi / 180.,
                    'opening_angle':np.pi / 2.,
                    'azimuth':116.5 * np.pi / 180.,
                    'duty_factor':0.85,
                    'detector_class':'earthL',
                    'plotrange':'10, 1000, 1e-25, 1e-20',
                    'fmin':8,
                    'fmax':1024,
                    'spacing':'geometric',
                    'df':1/4,
                    'npoints':5000
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



def gwfish_analysis(PATH_TO_YAML, PATH_TO_INJECTIONS, events_list, waveform, estimator,
                    detectors, fisher_parameters, calculate_errors = True, duty_cycle = False):

    for event in tqdm(events_list):

        population = '%s_BBH_%s' %(estimator, event)

        detectors_list = detectors[event]
        detectors_event = []
        for j in range(len(detectors_list)):
            detectors_event.append(detectors_list[j])
        networks = np.linspace(0, len(detectors_event) - 1, len(detectors_event), dtype=int)
        networks = str([networks.tolist()])

        detectors_ids = np.array(detectors_event)
        networks_ids = json.loads(networks)
        ConfigDet = os.path.join(PATH_TO_YAML + event + '.yaml')


        waveform_model = waveform
        waveform_class = gw.waveforms.LALFD_Waveform

        gw_parameters = pd.read_hdf(PATH_TO_INJECTIONS + event +  '/%s_%s_%s.hdf5' %(event, waveform, estimator))
        gw_parameters['mass1_lvk'] = gw_parameters['mass_1']
        gw_parameters['mass2_lvk'] = gw_parameters['mass_2']
        gw_parameters['mass_1'], gw_parameters['mass_2'] = from_mChirp_q_to_m1_m2(gw_parameters['chirp_mass'], gw_parameters['mass_ratio'])

        threshold_SNR = np.array([0., 1.])
        network = gw.detection.Network(detectors_ids, detection_SNR=threshold_SNR, parameters=gw_parameters,
                                    fisher_parameters=fisher_parameters, config=ConfigDet)
        k = 0
        parameter_values = gw_parameters.iloc[k]

        networkSNR_sq = 0
        for d in np.arange(len(network.detectors)):
            data_params = {
                'frequencyvector': network.detectors[d].frequencyvector,
                'f_ref': 50.
            }
            waveform_obj = waveform_class(waveform_model, parameter_values, data_params)
            wave = waveform_obj()
            t_of_f = waveform_obj.t_of_f

            signal = gw.detection.projection(parameter_values, network.detectors[d], wave, t_of_f)

            SNRs = gw.detection.SNR(network.detectors[d], signal, duty_cycle=duty_cycle)
            networkSNR_sq += np.sum(SNRs ** 2)
            network.detectors[d].SNR[k] = np.sqrt(np.sum(SNRs ** 2))

            if calculate_errors:
                network.detectors[d].fisher_matrix[k, :, :] = \
                    gw.fishermatrix.FisherMatrix(waveform_model, parameter_values, fisher_parameters, network.detectors[d], waveform_class=waveform_class).fm

        network.SNR[k] = np.sqrt(networkSNR_sq)

        gw.detection.analyzeDetections(network, gw_parameters, population, networks_ids)
        if calculate_errors:
            gw.fishermatrix.analyzeFisherErrors(network, gw_parameters, fisher_parameters, population, networks_ids)


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
    m2 = mChirp * (1 + q)**(1/5) * q**(3/5)
    return m1, m2

def derivative_m1_m2_dmChirp_dq(m1, m2, mChirp, q):
    """
    Compute the derivative of m1, m2 with respect to mChirp, q
    """
    dm1_dmChirp = (1 + q)**(1/5) * q**(-3/5)
    dm1_dq = mChirp * (1 + q)**(1/5) * (-3/5) * q**(-8/5) + mChirp * (1 + q)**(-4/5) * q**(-3/5)
    dm2_dmChirp = (1 + q)**(1/5) * q**(3/5)
    dm2_dq = mChirp * (1 + q)**(1/5) * (3/5) * q**(2/5) + mChirp * (1 + q)**(-4/5) * q**(3/5)
    return dm1_dmChirp, dm1_dq, dm2_dmChirp, dm2_dq


def jacobian_for_derivative_from_m1_m2_to_mChirp_q(m1, m2, fisher_matrix):
    """
    Compute the Jacobian for the transformation from m1, m2 to mChirp, q
    """
    mChirp, q = from_m1_m2_to_mChirp_q(m1, m2)
    dm1_dmChirp, dm1_dq, dm2_dmChirp, dm2_dq = derivative_m1_m2_dmChirp_dq(m1, m2, mChirp, q)
    rotated_fisher = fisher_matrix.copy()
    rotated_fisher[0, 0, 0] *= dm1_dmChirp**2
    rotated_fisher[0, 1, 1] *= dm2_dq**2
    rotated_fisher[0, 0, 1] *= dm1_dmChirp * dm2_dq
    rotated_fisher[0, 1, 0] *= dm1_dmChirp * dm2_dq
    rotated_fisher[0, 0, 2:] *= dm1_dmChirp
    rotated_fisher[0, 1, 2:] *= dm2_dq
    rotated_fisher[0, 2:, 0] *= dm1_dmChirp
    rotated_fisher[0, 2:, 1] *= dm2_dq

    #nparams = len(fisher_parameters)
    #jacobian = np.identity((nparams))

    #jacobian[np.ix_([fisher_parameters.index('mass_1'), fisher_parameters.index('mass_1')], [fisher_parameters.index('mass_1'), fisher_parameters.index('mass_1')])] = derivative_m1_m2_dmChirp_dq(m1, m2, mChirp, q)

    # Write the jacobian matrix to pass from the fisher matrix in m1 and m2 to fisher in mChirp and q
    #rotated_fisher = jacobian.T@old_fisher@jacobian
    return rotated_fisher

def get_rotated_fisher_matrix(PATH_TO_FISHERS, PATH_TO_RESULTS, events_list, detectors_list, estimator, lbs_signals, lbs_errs, new_fisher_parameters):
   
    for event in events_list:
        signals = pd.read_csv(PATH_TO_RESULTS + 'results/gwfish_m1_m2_from_mChirp_q/signals/' +
                            'Signals_%s_BBH_%s.txt' %(estimator, event), names = lbs_signals, skiprows = 1,
                            delimiter = ' ')
        detectors_labels = list(detectors_list[event])
        connector = '_'
        label = detectors_labels[0]
        for j in range(1, len(detectors_labels)):
            label += connector + detectors_labels[j]

        fishers = np.load(PATH_TO_RESULTS + 'results/gwfish_m1_m2_from_mChirp_q/fishers/' + 
                          'Fishers_%s_%s_BBH_%s_SNR1.0.npy' %(label, estimator, event))
        m1, m2 = signals[['mass_1', 'mass_2']].iloc[0]
        rotated_fisher = jacobian_for_derivative_from_m1_m2_to_mChirp_q(m1, m2, fishers)
        np.save(PATH_TO_RESULTS + 'results/gwfish_m1_m2_from_mChirp_q/rotated_fishers/' + 
                'Rot_Fishers_%s_%s_BBH_%s_SNR1.0.npy' %(label, estimator, event), rotated_fisher)

        inv_rotated_fisher, sing_values = gw.fishermatrix.invertSVD(rotated_fisher[0, :, :])
        np.save(PATH_TO_RESULTS + 'results/gwfish_m1_m2_from_mChirp_q/rotated_inv_fishers/' + 
                'Rot_Inv_Fishers_%s_%s_BBH_%s_SNR1.0.npy' %(label, estimator, event), inv_rotated_fisher)
        
        
        old_errors = pd.read_csv(PATH_TO_RESULTS + 'results/gwfish_m1_m2_from_mChirp_q/errors/' + 
                                 'Errors_%s_%s_BBH_%s_SNR1.0.txt' %(label, estimator, event), names = lbs_errs, 
                                 skiprows = 1, delimiter = ' ')
        new_errors = old_errors.copy()

        err_params = []
        for l in range(len(new_fisher_parameters)):
            err_params.append('err_' + new_fisher_parameters[l])
        new_errors[err_params] = np.sqrt(np.diag(inv_rotated_fisher))
        np.savetxt(PATH_TO_RESULTS + 'results/gwfish_m1_m2_from_mChirp_q/rotated_errors/' +
                'Errors_%s_%s_BBH_%s_SNR1.0.txt' %(label, estimator, event), new_errors, delimiter = ' ', 
                fmt = '%.15f', header = '# ' + ' '.join(new_errors.keys()), comments = '')


def get_samples_from_MVN(means, cov, N):
    """
    Draw samples from a multivariate normal distribution
    """
    return np.random.multivariate_normal(means, cov, N)

def get_samples_from_TMVN(min_array, max_array, means, cov, N):
    """
    Draw samples from a truncated multivariate normal distribution
    """
    tmvn = TruncatedMVN(means, cov, min_array, max_array)
    return tmvn.sample(N)

def get_posteriors(samples, params, priors_dict, min_array, max_array, N):
    """
    Draw samples from a multivariate normal distribution with priors
    """
    samples['priors'] = np.ones_like(samples[params[0]])
    for param in params:
        samples['priors'] *= priors_dict[param](samples[param], lower_bound = min_array[param], upper_bound = max_array[param])

    samples['weights'] = samples['priors'] / np.sum(samples['priors'])
    prob = samples['weights'].to_numpy()
    index = np.random.choice(np.arange(N), size = N, replace = True, p = prob)
    posteriors = samples.iloc[index]
    
    return posteriors

def get_lvk_samples():



