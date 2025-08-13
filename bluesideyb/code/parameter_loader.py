import json
import numpy as np
from scipy.constants import c, h, physical_constants

# Load JSON
with open('yb_constants.json') as f:
    data = json.load(f)

# Speed of light in m/s
c_m_s = c  # ≈2.99792458e8

def wl_to_Hz(wavelength_nm):
    """Convert vacuum wavelength in nm to frequency in Hz."""
    return c_m_s / (wavelength_nm * 1e-9)

# Process constants
params = {}

# Qubit splitting
hf = data['levels']['S1/2']['hyperfine_qubit']['splitting_GHz']
params['hf_splitting_Hz'] = hf * 1e9

# Optical transitions
for level in ['P1/2', 'P3/2']:
    lvl = data['levels'][level]
    lam = lvl.get('wavelength_nm')
    if lam:
        params[f'{level}_freq_Hz'] = wl_to_Hz(lam)

# Trap & laser
t = data['trap_and_laser_parameters']
params['trap_freq_Hz'] = t['trap_frequency_motional_MHz'] * 1e6
params['lamb_dicke'] = t['lamb_dicke_parameter']
params['raman_detuning_Hz'] = t['raman_detuning_GHz'] * 1e9
params['raman_rabi_Hz'] = t['raman_rabi_frequency_kHz'] * 1e3
params['microwave_rabi_Hz'] = t['microwave_rabi_frequency_kHz'] * 1e3
params['laser_linewidth_Hz'] = t['laser_linewidth_Hz']
params['motional_heating_quanta_per_s'] = t['motional_heating_rate_quanta_per_ms'] * 1e3

# Decay rates
for lvl in ['P1/2', 'P3/2', 'D3/2', 'D5/2', 'F7/2']:
    rate = data['levels'][lvl].get('decay_rate_s')
    if rate:
        params[f'{lvl}_decay_rate_s'] = rate

# Branching details (example)
p1 = data['levels']['P1/2']['branching']
params['P1_to_S_rate'] = p1['to_S1/2']['rate_s']
params['P1_to_D3_rate'] = p1['to_D3/2']['rate_s']

# Dump for confirmation
import pprint; pprint.pprint(params)
