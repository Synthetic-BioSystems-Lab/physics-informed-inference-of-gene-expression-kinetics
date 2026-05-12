# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:45:45 2026

@author: zacha
"""

from biocrnpyler import *
import numpy as np
import matplotlib.pyplot as plt

class bioCRNpyler_sim():
    def __init__(self, timepoints, D, ktx=0.05, ktl=0.05, kdil=0.0075):
        
        # Parameters and Global Mechanisms
        parameters={"ktx":ktx, "ktl":ktl, "kdeg":0.001, "kdil":kdil}

        dilution_mechanism = Dilution(filter_dict = {"degtagged":True}, default_on = False)
        global_mechanisms = {"dilution":dilution_mechanism}
        
        # DNA parts
        prom = Promoter('prom')
        rbs = RBS('rbs')
        CDS_YFP = CDS('YFP', 'YFP')
        CDS_YFP.protein = Species('YFP', material_type='protein', attributes=['degtagged'])
        t = Terminator('t')
        YFP_construct = DNA_construct(name='dna_YFP', parts_list=[prom, rbs, CDS_YFP, t])
        
        # Mixture and CRN creation
        M = SimpleTxTlExtract('simtxtl', parameters = parameters, global_mechanisms=global_mechanisms, 
                              components=[YFP_construct])
        CRN = M.compile_crn()
        CRN.write_sbml_file(f'crn_docs/test.xml') #saving CRN as sbml
        
        with open('crn_docs/temp_CRN_EQNs.txt', 'w') as f:
            f.write(CRN.pretty_print(show_rates = True, show_keys = True))
        
        # Simulation and Plotting  
        x0 = {YFP_construct.get_species():D}
        self.Res = CRN.simulate_with_bioscrape_via_sbml(timepoints, initial_condition_dict = x0)
        
    def get_YFP(self):
        return self.Res['protein_YFP_degtagged']
    def get_final_mRNA(self):
        return self.Res['rna_part_rbs_forward_part_YFP_forward_part_t_forward_'].iloc[-1]
    def get_time(self):
        return self.Res['time']
    
#Data Generation
timepoints = np.linspace(0, 10000, 100)
X_lst, Y_lst = [], []

plt.figure()
    
for i in range(100):
    
    ktl = abs(np.random.uniform(0.01, 0.1))# + np.random.normal(scale=0.005))
    kdil = abs(np.random.uniform(0.001, 0.01))# + np.random.normal(scale=0.0005))
    D = abs(1)
    sim = bioCRNpyler_sim(timepoints, D, ktl=ktl, kdil=kdil)
    
    X_lst.append(sim.get_YFP())
    Y_lst.append([ktl, kdil, sim.get_final_mRNA()])
    plt.plot(sim.get_time(), sim.get_YFP(), alpha=0.3, color='g')
    

plt.xlabel('Time')
plt.ylabel('[FP]')
plt.show()

# Y_array = np.array(Y_list)     
np.save('sim_TU_data/yfp.npy', X_lst)
np.save('sim_TU_data/param_labels.npy', Y_lst)
