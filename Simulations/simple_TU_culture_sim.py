from biocrnpyler import *
from bioscrape.types import Model
from bioscrape.simulator import py_simulate_model
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def simpleTUsbmlGenerator(ktx=0.05,ktl=0.05, kdil=0.0075):
    # Parameters and Global Mechanisms
    parameters={"ktx":ktx, "ktl":ktl, "kdil":kdil}

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
    CRN.write_sbml_file(f'Simulations/crn_docs/temp.xml') #saving CRN as sbml

class StateContainer:
    def __init__(self, initial_state:dict):
        self.state = initial_state
        self.df = pd.DataFrame([initial_state])

    def update_state_var(self, var:str, value:float):
        self.state[var] = value

    def get_state_var(self, var:str):
        return self.state[var]
    
    def append_df(self):
        temp_df = pd.DataFrame([self.state])
        self.df = pd.concat([self.df, temp_df], ignore_index=True)

    def get_df(self):
        return self.df
    
class GrowthModel:
    def __init__(self,statecontain:StateContainer, k_growth=0.000133, K=1.2e9, t_lag=25, method='LSODA', dt=600):
        self.statecontain = statecontain
        self.method = method
        self.k_growth = k_growth
        self.K = K
        self.t_lag = t_lag
        self.dt = dt

    def growth_eqn(self, t, X):

        # #logistic growth model 
        # if t > self.t_lag:
        #     dX = self.k_growth*X*(1 - X/self.K)
        # else:
        #     dX = 0
        # return dX

        #gompertz growth model
        if t > self.t_lag:
            dX = self.k_growth * np.log(self.K/X) * X
        else:
            dX = 0
        return dX
    
    def run_sim(self, t_span):
        sol = solve_ivp(lambda t, X: self.growth_eqn(t, X), t_span, [self.statecontain.get_state_var("X")], method=self.method)
        sol_df = pd.DataFrame({"time": sol.t, "X": sol.y[0]})
        return sol_df

class bioscrapeModel:
    def __init__(self, sbml_filename:str, statecontain:StateContainer, dt=600):
        self.statecontain = statecontain
        self.model = Model(sbml_filename=sbml_filename)
        self.dt = dt

    def run_sim(self, t_span, num_time_step=10):
        self.model.set_species(self.statecontain.state)
        timepoints = np.linspace(t_span[0], t_span[1], num_time_step)
        res = py_simulate_model(timepoints, self.model, stochastic=False)
        return res
    
class Multimodel:
    def __init__(self, models:list, state_container:StateContainer, t_final=200):
        self.models = models
        self.state_container = state_container
        self.t_final = t_final

    def run_sim(self):
        t = 0
        while t < self.t_final:
            for model in self.models:
                for key, value in model.run_sim([t, t+model.dt]).items():
                    self.state_container.update_state_var(key, value.iloc[-1])
                    #print(f"Updated {key} to {value.iloc[-1]} at time {t}")

            self.state_container.append_df()
            self.state_container.update_state_var("dna_part_prom_forward_part_rbs_forward_part_YFP_forward_part_t_forward_", 
                                                  self.state_container.get_state_var("X"))
            t += model.dt

        return self.state_container.get_df()

if __name__ == "__main__":

    np.random.seed(8380)

    X_lst = []
    Y_lst = []
    Z_lst = []
    cell_lst = []
    mrna_lst = []

    plt.figure()

    for i in range(1000):

        ktx = 0.05
        ktl = abs(np.random.uniform(0.01, 0.1))# + np.random.normal(scale=0.005))
        kdil = abs(np.random.uniform(0.001, 0.01))# + np.random.normal(scale=0.0005))
        tlag = 25 # = abs(np.random.uniform(5*3600))# + np.random.normal(scale=5))

        simpleTUsbmlGenerator(ktx=ktx, ktl=ktl, kdil=kdil)
        state_container = StateContainer(initial_state={"X":1, #1.72e6, 
                                                    "dna_part_prom_forward_part_rbs_forward_part_YFP_forward_part_t_forward_":1, #1.72e6,
                                                    "protein_YFP_degtagged":0,
                                                    "rna_part_rbs_forward_part_YFP_forward_part_t_forward_":0
                                                    })
        growth_model = GrowthModel(statecontain=state_container, t_lag=tlag, k_growth=0.05, K=500, dt=10)
        bioscrape_model = bioscrapeModel(sbml_filename="Simulations/crn_docs/temp.xml", statecontain=state_container, dt=10)
        models = [growth_model, bioscrape_model]
        multimodel = Multimodel(models=models, state_container=state_container, t_final=500)#=72000)
        multimodel_res = multimodel.run_sim()
        plt.plot(multimodel_res["time"]/3600, multimodel_res["protein_YFP_degtagged"], alpha=0.3, color='g')

        cell_lst.append(multimodel_res["X"])
        X_lst.append(multimodel_res["protein_YFP_degtagged"])
        mrna_lst.append(multimodel_res["rna_part_rbs_forward_part_YFP_forward_part_t_forward_"])
        #print(multimodel_res['rna_part_rbs_forward_part_YFP_forward_part_t_forward_'].iloc[-1])
        Y_lst.append([ktl, kdil, multimodel_res['rna_part_rbs_forward_part_YFP_forward_part_t_forward_'].iloc[-1]])
        Z_lst.append([ktx, kdil, ktl, kdil])


    plt.show()

    np.save('Simulations/sim_TU_data/cell_conc_culture.npy', cell_lst)
    np.save('Simulations/sim_TU_data/yfp_culture.npy', X_lst)
    np.save('Simulations/sim_TU_data/mrna_culture.npy', mrna_lst)
    np.save('Simulations/sim_TU_data/param_labels_culture.npy', Y_lst)
    np.save('Simulations/sim_TU_data/time_culture.npy', multimodel_res['time']/3600)
    np.save('Simulations/sim_TU_data/param_labels_culture_PIAE.npy', Z_lst)
