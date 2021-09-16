import os
import sys
import re
import logging
import time
import copy
import json

import subprocess

import numpy as np
import pandas as pd
import torch

from padelpy import from_mdl, padeldescriptor

from binary_classifier import *

WORKDIR = './workdir'

if not os.path.isdir(WORKDIR):
    os.mkdir(WORKDIR)

MOPAC_COMMAND_OPT = 'OPT PM6'
MOPAC_COMMAND_FORCE = 'FORCE PM6'
MOPAC_COMMAND_POLAR = 'PM6 POLAR'
MOPAC_COMMAND_INDOS_S = 'INDO CIS C.A.S.=(2,1) TDIP MAXCI=50 SINGLET'
MOPAC_COMMAND_INDOS_T = 'INDO CIS C.A.S.=(2,1) TDIP MAXCI=50 TRIPLET'

##########################!!!WARNING!!!###########################
# THE DESCRIPTORS MUST BE ORDERED THE SAME WAY THEY WERE ORDERED #
# DURING THE NN-TRAINING!!!                                      #
##################################################################

NN_QC_DESC = ['E_LUMO_PM6[eV]', 'IsotropicAverageAlpha [A.U.]', 'AverageGamma [A.U.]', 'S1(CAS2-1+CIS) [eV]', 'INDOS-OscStr(CAS2-1+CIS)']

CHEMOMETRY_DESC = ['Si', 'Mv', 'Mare', 'Mi', 'nHBAcc', 'nHBDon', 'n6Ring', 'n8HeteroRing', 'nF9HeteroRing', 'nT5HeteroRing', 'nT6HeteroRing', 'nT7HeteroRing', 'nT8HeteroRing', 'nT9HeteroRing', 'nT10HeteroRing', 'nRotB', 'RotBFrac', 'nRotBt', 'RotBtFrac']

MEAN_STD_FILE_PATH = 'MeanStd20210905-234006.json'
TRAINED_NN_FILE_PATH = 'TrainedANN20210905-234006'

def loginit(logger):
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class PropertiesCalculation:
    CSV_NAMES_TO_ATTRIBUTES = {
            'E_HOMO_PM6[eV]':'e_homo',
            'E_LUMO_PM6[eV]':'e_lumo',
            'DipoleMoment [D]':'dipole_moment',
            'IsotropicAverageAlpha [A.U.]':'alpha',
            'AverageBeta [A.U.]':'beta', 
            'AverageGamma [A.U.]':'gamma',
            'S1(CAS2-1+CIS) [eV]':'S1',
            'S2(CAS2-1+CIS) [eV]':'S2',
            'INDOS-OscStr(CAS2-1+CIS)':'oscillator_strength',
            'CI-Coef**2(CAS2-1+CIS)':'CI_coef',
            'T1(CAS2-1+CIS) [eV]':'T1',
            'T2(CAS2-1+CIS) [eV]':'T2'
            }

    def __init__(self, inp_mdl_str):
        self.timestr = time.strftime('%Y%m%d-%H%M%S')# used in file naming
        self.inp_mdl_str = inp_mdl_str
        self.logger = logging.getLogger(
                      'Molecular Property Calculator')
        loginit(self.logger)
        self.inp_mdl_name = 'input{}.mdl'.format(self.timestr)
        self.inp_mdl_name = os.path.join(WORKDIR, self.inp_mdl_name)
        with open(self.inp_mdl_name, 'w') as f:
            f.write(self.inp_mdl_str)
        # Molecule properties in format (value, unit)
        self.e_homo = (None, None)
        self.e_lumo = (None, None)
        self.dipole_moment = (None, None)
        self.alpha = (None, None)
        self.beta = (None, None)
        self.gamma = (None, None)
        self.S1 = (None, None)
        self.S2 = (None, None)
        self.oscillator_strength = (None, None)
        self.CI_coef = (None, None)
        self.T1 = (None, None)
        self.T2 = (None, None)
        self.chemometry_descriptors = {}
        self.drc_ann_prediction = None

    def run_mopac_single_point(self, 
                               mopac_inp_file_name,
                               mopac_command, 
                               atoms, 
                               coord):
        self.logger.info('Generating MOPAC file: {}'.format(mopac_inp_file_name))
        with open(mopac_inp_file_name, 'w') as f:
            f.write('{}\n\n\n'.format(mopac_command))
            for i, atom in enumerate(atoms):
                f.write('{:>5}'.format(atom))
                for c in coord[i]:
                    f.write('{:15.9f}'.format(c))
                f.write('\n')
        
        self.logger.info('Running Mopac calculation: {}'.format(mopac_inp_file_name))
        subprocess.run(
            ['/opt/mopac/MOPAC2016.exe', 
            mopac_inp_file_name],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL
            )

    
    def geom_opt(self):
        self.logger.info('Running OpenBabel to obtain the initial geometry')
        # MOPAC input file preparation
        p1 = subprocess.Popen(
                ['echo', self.inp_mdl_str],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL)
        p2 = subprocess.Popen(
                 ['babel', '-imdl', '-omop'],
                 stdin = p1.stdout,
                 stdout = subprocess.PIPE,
                 stderr = subprocess.DEVNULL)
        p1.stdout.close()
        result = p2.communicate()[0]
        #TODO: Raise error if the command exit code != 0 !!!
        result = result.decode('utf-8')
        result = re.sub('PUT KEYWORDS HERE', MOPAC_COMMAND_OPT, result)
        mopac_inp_file_name = 'opt_geom{}.dat'.format(self.timestr)
        mopac_inp_file_name = os.path.join(WORKDIR, mopac_inp_file_name)
        with open(mopac_inp_file_name, 'w') as f:
            f.write(result)
        # MOPAC Geometry Optimization
        subprocess.run(
            ['/opt/mopac/MOPAC2016.exe', 
            mopac_inp_file_name],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL)
        #TODO: Check if mopac ran successfuly
        with open(mopac_inp_file_name[:-3]+'out', 'r') as f:
            opt_geom_data = f.read()
       # self.number_of_atoms = int(re.search(
       #         r'(?<= =) (.*) (?=atoms)', 
       #         opt_geom_data).group(0))
        atoms_coord = re.search(r'(?<= CARTESIAN COORDINATES\n\n)(.*?)(?=\n\n)',
                opt_geom_data, re.DOTALL)
        #TODO Check if atoms coord is obtained, else raise exception!
        atoms_coord = atoms_coord.group(0)
        atoms, coord = [], []
        for line in atoms_coord.split('\n'):
            line_lst = line.split()
            atoms.append(line_lst[1])
            coord.append([float(_) for _ in line_lst[2:]])
        return atoms, np.array(coord)

    def check_imag_freq(self, atoms, coord):
        mopac_inp_file_name = 'force{}.dat'.format(self.timestr)
        self.logger.info('Checking for imaginary frequencies: {}'.\
                format(mopac_inp_file_name))
        mopac_inp_file_name = os.path.join(WORKDIR, mopac_inp_file_name)
        self.run_mopac_single_point(
                mopac_inp_file_name,
                MOPAC_COMMAND_FORCE, 
                atoms, 
                coord)
        
        with open(mopac_inp_file_name[:-3]+'out', 'r') as f:
            force_calc_data = f.read()

        if re.search('IMAGINARY', force_calc_data):
            self.logger.warn('Imaginary frequencies in: {}'.\
                    format(mopac_inp_file_name))
            return True
        else:
            return False
    
    def run_mopac_polar(self, atoms, coord):
        mopac_inp_file_name = 'polar{}.dat'.format(self.timestr)
        self.logger.info('Running MOPAC polar caculation: {}'.\
                format(mopac_inp_file_name))
        mopac_inp_file_name = os.path.join(WORKDIR, mopac_inp_file_name)
        self.run_mopac_single_point(
                mopac_inp_file_name,
                MOPAC_COMMAND_POLAR, 
                atoms, 
                coord)

        self.logger.info('MOPAC polar calculation done')
        with open(mopac_inp_file_name[:-3]+'out', 'r') as f:
            polar_calc_data = f.read()
        
        # Getting HOMO and LUMO energies
        #TODO add error handling if regex not found
        n_doubly_occ_states = re.search(
                '(?<= RHF CALCULATION, NO. OF DOUBLY OCCUPIED LEVELS =)(.*)',
                polar_calc_data)
        n_doubly_occ_states = int(n_doubly_occ_states.group(0))
        energies = re.search(
                r'(?<=EIGENVALUES)(.*?)(?=\n\n)',
                polar_calc_data, re.DOTALL)
        energies = energies.group(0).strip()
        energies = energies.split()
        self.e_homo = (float(energies[n_doubly_occ_states-1]), 'eV')
        self.e_lumo = (float(energies[n_doubly_occ_states]), 'eV')
        dipole_moment_str = re.search('SUM.*', polar_calc_data).group(0)
        #TODO add error hanler for the float(...)
        self.dipole_moment = (
                float(dipole_moment_str.strip().split()[-1]),
                'D')
        alpha_str = re.search(
                '(?<=ISOTROPIC AVERAGE ALPHA =)(.*)',
                polar_calc_data).group(0)
        self.alpha = (
                float(alpha_str.strip().split()[0]),
                'A.U.')
        beta_str = re.search(
                '(?<=AVERAGE BETA \(SHG\) VALUE AT)(.*)',
                polar_calc_data).group(0)
        self.beta = (
                float(beta_str.strip().split()[3]),
                'A.U.')
        gamma_str = re.search(
                '(?<= AVERAGE GAMMA VALUE AT)(.*)',
                polar_calc_data).group(0)
        self.gamma = (
                float(gamma_str.strip().split()[2]),
                'A.U.')
    
    def run_mopac_indos(self, atoms, coord):
        mopac_inp_file_name_s = 'cas_singlet{}.dat'.format(self.timestr)
        self.logger.info('Running MOPAC CIS CAS (2,1) siglet caculation: {}'.\
                format(mopac_inp_file_name_s))
        mopac_inp_file_name_s = os.path.join(WORKDIR, mopac_inp_file_name_s)
        self.run_mopac_single_point(
                mopac_inp_file_name_s,
                MOPAC_COMMAND_INDOS_S, 
                atoms, 
                coord)
        
        mopac_inp_file_name_t = 'cas_triplet{}.dat'.format(self.timestr)
        self.logger.info('Running MOPAC CIS CAS (2,1) triplet caculation: {}'.\
                format(mopac_inp_file_name_t))
        mopac_inp_file_name_t = os.path.join(WORKDIR, mopac_inp_file_name_t)
        self.run_mopac_single_point(
                mopac_inp_file_name_t,
                MOPAC_COMMAND_INDOS_T, 
                atoms, 
                coord)

        with open(mopac_inp_file_name_s[:-3]+'out', 'r') as f:
            cas_calc_data_s = f.readlines()
        for i, line in enumerate(cas_calc_data_s):
            if 'CI trans.  energy frequency wavelength oscillator' in line:
                break
        self.S1 = (float(cas_calc_data_s[i+3].split()[1]), 'eV')
        self.S2 = (float(cas_calc_data_s[i+4].split()[1]), 'eV')
        self.oscillator_strength = (float(cas_calc_data_s[i+3].split()[4]), '')
        
        for i, line in enumerate(cas_calc_data_s):
            if ' CI excitations=' in line:
                break
        conf = 0
        for line in cas_calc_data_s[(i+5):]:
            if line.strip() == '':
                break
            line_lst = line.strip().split()
            if int(line_lst[7]) == 2:
                conf = int(line_lst[0])
                break
        for i, line in enumerate(cas_calc_data_s):
            if 'State    1' in line:
                break
        
        for line in cas_calc_data_s[(i+1):]:
            if 'Total coeff printed' in line:
                break
            line_lst = line.split()
            if int(line_lst[1]) == conf:
                self.CI_coef = (float(line_lst[3]), '')
        self.CI_coef = (0.0, '')
        
        with open(mopac_inp_file_name_t[:-3]+'out', 'r') as f:
            cas_calc_data_t = f.readlines()
        for line in cas_calc_data_s:
            if 'Depression of ground-state after' in line:
                S0 = float(line.strip().split()[-2])
        for line in cas_calc_data_t:
            if 'Depression of ground-state after' in line:
                T1 = float(line.strip().split()[-2])
        self.T1 = ((T1 - S0), 'eV')
        for i, line in enumerate(cas_calc_data_t):
            if 'CI trans.  energy frequency wavelength oscillator' in line:
                break
        self.T2 = (float(cas_calc_data_t[i+3].split()[1]) + self.T1[0], 'eV')
 

    def get_chemometry_descriptors(self):
       self.logger.info('Generating PaDEL descriptors')
       padeldescriptor(d_3d = False, d_2d = True)
       out_csv = os.path.join(
                  WORKDIR, 
                  'chemo_desc{}.csv'.format(self.timestr))
       from_mdl(self.inp_mdl_name, output_csv=out_csv)
       df = pd.read_csv(out_csv)
       df = df[CHEMOMETRY_DESC]
       self.chemometry_descriptors = df.to_dict('records')[0]
 
    def read_mean_std(self, mean_std_fname):
        self.logger.info(
                'Reading descriptor training set mean and standard\
                 deviation values from {}'.format(mean_std_fname))
        with open(mean_std_fname, 'r') as f:
            mean_std = json.load(f)
        return mean_std

    def attributes_to_torch_tensor(self, standardize = True):
        d = copy.deepcopy(self.__dict__)
        del d['inp_mdl_str']
        del d['logger']
        del d['inp_mdl_name']
        del d['timestr']
        #print(d)
        del d['chemometry_descriptors']
        del d['drc_ann_prediction']
        
        if standardize:
            mean_std = self.read_mean_std(self.abs_path(MEAN_STD_FILE_PATH))

        descriptors = []
        for _desc in NN_QC_DESC:
            #TODO Add try ... except to make sure that _desc makes sense
            item = self.CSV_NAMES_TO_ATTRIBUTES[_desc]
            if d[item] == (None, None) or \
                        d[item] == None:
                self.logger.error('{} should not be None'.format(item))
                raise ValueError('{} should not be None'.format(item))
            else:
                if standardize:
                    val = (d[item][0] - mean_std['mean'][_desc])/mean_std['stdev'][_desc]
                else:
                    val = d[item][0]
                descriptors.append(val)
        if self.chemometry_descriptors == {}:
               self.logger.error('{}.chemometry_descriptors should not be None'.format(self.__name__))
               raise ValueError('{}.chemometry_descriptors should not be None'.format(self.__name__))
        else:
            for item in self.chemometry_descriptors:
                if standardize:
                    val = (self.chemometry_descriptors[item] - mean_std['mean'][item])/mean_std['stdev'][item]
                else:
                    val = self.chemometry_descriptors[item]
                descriptors.append(val)
        
        return torch.tensor(descriptors, dtype=torch.float)
    
    def run_ann(self, input_tensor):
        self.logger.info('Loading ANN model from {}'.format(TRAINED_NN_FILE_PATH))
        model = torch.load(self.abs_path(TRAINED_NN_FILE_PATH))
        model.eval()
        self.logger.info('Running ANN model')
        self.drc_ann_prediction = model(input_tensor)
    
    def run_calculations(self):
        a, c = copy.deepcopy(self.geom_opt())
        #self.check_imag_freq(a,c)
        self.run_mopac_polar(a,c)
        self.run_mopac_indos(a,c)
        self.get_chemometry_descriptors()
        nn_descs = self.attributes_to_torch_tensor()
        self.run_ann(nn_descs)
        d = self.__dict__
        del d['inp_mdl_str']
        del d['logger']
        del d['inp_mdl_name']
        del d['drc_ann_prediction']
        return d 
       
    def abs_path(self, file_name):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name)
    
if __name__ == '__main__':
    #with open(sys.argv[1], 'r') as f:
    #    data = f.read()
    data = sys.argv[1]
    p = PropertiesCalculation(data)
    print(p.run_calculations())
