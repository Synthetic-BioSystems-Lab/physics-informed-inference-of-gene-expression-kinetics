import numpy as np
import simple_TU_culture_sim as stucs

ktx = 0.05
ktl = 0.05
kdil = 0.005
tlag = 25

stucs.simpleTUsbmlGenerator(ktx=ktx, ktl=ktl, kdil=kdil)
state_container = stucs.StateContainer(initial_state={"X":1, #1.72e6, 
                                                "dna_part_prom_forward_part_rbs_forward_part_YFP_forward_part_t_forward_":1, #1.72e6,
                                                "protein_YFP_degtagged":0,
                                                "rna_part_rbs_forward_part_YFP_forward_part_t_forward_":0
                                                })
growth_model = stucs.GrowthModel(statecontain=state_container, t_lag=tlag, k_growth=0.05, K=500, dt=10)
bioscrape_model = stucs.bioscrapeModel(sbml_filename="Simulations/crn_docs/temp.xml", statecontain=state_container, dt=10)
models = [growth_model, bioscrape_model]
multimodel = stucs.Multimodel(models=models, state_container=state_container, t_final=500)#=72000)
multimodel_res = multimodel.run_sim()

P = multimodel_res["protein_YFP_degtagged"].to_numpy()
mrna = multimodel_res["rna_part_rbs_forward_part_YFP_forward_part_t_forward_"].to_numpy()
time_lst = multimodel_res["time"].to_numpy()
dt = time_lst[-1] - time_lst[-2]

# dP = np.zeros_like(P)
dP = np.gradient(P, dt)

print(dP.shape)
print(mrna.shape)

# ktl = np.expand_dims(ktl, 1)
# kdil = np.expand_dims(kdil, 1)

# 0 = (ktl * M) - (kdil * A) - dAdt
eps = 1e-6
res = ktl * mrna - kdil * P - (dP)
scale1 = (ktl * mrna) + (kdil * P) + dP + eps
loss_phys = (res / scale1)

print(res)
print(loss_phys)
