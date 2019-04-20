"""
TASC_learning,py

Learning TASC architecture for SI swarm
"""

import numpy as np
import _pickle as pkl
import matplotlib.pyplot as plt

from objects.HierarchicalEncoder import HierarchicalEncoder
from objects.DoubleIntegratorDynamics import DoubleIntegratorDynamics
from objects.BasicPol import BasicPol_Swarm_AbsTrack, BasicPol
from objects.Shooting_AbsTrack import Shooting_AbsTrack
from objects.MFRMFP import MFRMFP_Plane, MFRMFP_Swarm_SI

import exp_SI1 as exp

n_epochs= 10
n_batches = 1000
batch_size = 128
val_size = 512

if __name__ == '__main__':
    print('Setting up Dynamics...')
    prt_dyn = MFRMFP_Plane(params=exp.prnt_dyn_params, home=exp.home, name='Embedding_Plane_Dynamics')
    swm_dyn = MFRMFP_Swarm_SI(params=exp.swrm_dyn_params, home=exp.home, name='Embedding_Swarm_Dynamics')

    print('Setting up Networks...')
    abs_enc = HierarchicalEncoder(exp.abst_state_size, exp.chld_state_size, exp.chld_param_size, params=exp.abst_enc_params, home=exp.home, name='abst_enc')
    abs_dyn = DoubleIntegratorDynamics(exp.abst_state_size, exp.prnt_state_size, exp.prnt_param_size, params=exp.abst_dyn_params, home=exp.home, name='abst_dyn')
    prt_pol = BasicPol((exp.prnt_state_size,), (exp.abst_state_size,), (exp.prnt_param_size,), params=exp.prnt_pol_params, home=exp.home, name='prnt_pol')
    swm_pol = BasicPol_Swarm_AbsTrack((None, exp.chld_state_size), (None, exp.chld_actin_size), abs_enc.abs_size, (None, exp.chld_param_size), params=exp.swrm_pol_params, home=exp.home, name='swrm_pol')

    print('Building Learner...')
    learner = Shooting_AbsTrack(abs_enc, abs_dyn, prt_pol, swm_pol, prt_dyn, swm_dyn, params=exp.full_trn_params, home=exp.home, name='learner_shooting')

    print('Training Networks...')
    hist = learner.train(n_epochs, n_batches, batch_size, val_size)
