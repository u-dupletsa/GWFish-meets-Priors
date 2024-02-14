import numpy as np
import pandas as pd
import h5py
import os
import pesummary
from pesummary.io import read
import pickle
import yaml


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
    dict_file = {'L1':{'lat':30.56 * np.pi / 180.,
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

        local_dictionary = dict_file

        data = h5py.File(PATH_TO_DATA + event + '.h5', 'r')
        detectors[event] = keys(data['C01:' + waveform]['psds'])

        for j in range(len(detectors[event])):
            local_dictionary[detectors[event][j]] = {'psd_data':'PSDs4GWFish/%s/psd_%s_%s.txt' %(waveform, event, detectors[event][j])}

            np.savetxt(PATH_TO_PSD + 'psd_%s_%s_%s.txt' %(waveform, event, detectors[event][j]), 
                       np.c_[data['C01:' + waveform]['psds'][detectors[event][j]][:, 0], 
                             data['C01:' + waveform]['psds'][detectors[event][j]][:, 1]])
                                                               
        with open(PATH_TO_YAML + '%s.yaml' %event, 'w') as my_yaml_file:
            yaml.dump(local_dictionary, my_yaml_file)

    
    with open(PATH_TO_RESULTS + 'info/' + 'detectors_dictionary.pkl', 'wb') as f:
        pickle.dump(detectors, f)



def gwfish_analysis()
