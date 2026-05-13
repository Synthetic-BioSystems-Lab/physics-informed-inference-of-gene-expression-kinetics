from bioscrape.lineage import LineageModel, LineageVolumeSplitter #For building our Model
from bioscrape.lineage import py_SimulateCellLineage
import numpy as np
import pylab as plt
import time as pytime
from time import process_time 

#Load Toggle CRN

vsplit_options = {
    "default":"binomial",
    "volume":"binomial",
    "dna_part_prom_forward_part_rbs_forward_part_YFP_forward_part_t_forward_":"duplicate"
}
copy_number = 1
x0 = {"dna_part_prom_forward_part_rbs_forward_part_YFP_forward_part_t_forward_": copy_number}
Msimple_TU_events = LineageModel(sbml_filename = "crn_docs/test.xml", initial_condition_dict = x0)

print("Species in Msimple_TU_events", list(Msimple_TU_events.get_species()))

g = .005

kgrow = 0.5

vsplit = LineageVolumeSplitter(Msimple_TU_events, options = vsplit_options)
#Msimple_TU_events.create_division_rule("deltaV", {"threshold":1.0}, vsplit)

Msimple_TU_events.create_volume_event("linear volume", {"growth_rate":g}, "massaction", {"k":kgrow, "species":""})

Msimple_TU_events.create_division_event("division", {}, "massaction", {"k":kgrow/1000., "species":""}, vsplit)

kdeath = 10
Kdeath = 1000
Msimple_TU_events.create_death_event("death", {}, "hillpositive", {"k":kdeath, "s1":"X", "n":2, "K":Kdeath})

timepoints = np.arange(0, 3000, 1.0)
print("Simulating")
ts = process_time()
lineage = py_SimulateCellLineage(timepoints = timepoints, Model = Msimple_TU_events)
te = process_time()
print("Simulation C=complete in", te-ts, "s")

sch_tree = lineage.get_schnitzes_by_generation()

color_list = [(i/len(sch_tree), 0, 1.-i/len(sch_tree)) for i in range(len(sch_tree))]

plt.figure(figsize = (10, 10))

plt.subplot(311)
YFP = np.zeros_like(timepoints)
cell_count = np.zeros_like(timepoints)
start = 0

for generation in range(len(sch_tree)):

    L = sch_tree[generation]

    for sch in L:

        df = sch.py_get_dataframe(Model = Msimple_TU_events)
        plt.plot(df["time"], df["protein_YFP_degtagged"], color = color_list[generation], alpha = .5)
        start = int(df["time"].iloc[0])
        first_index = df.index[0]

        if df["time"].iloc[-1] == timepoints[-1]:
            stop = int(df["time"].iloc[-1]) + 1
            last_index = df.index[-1] + 1
        else:
            stop = int(df["time"].iloc[-1])
            last_index = df.index[-1]

        YFP[start:stop] += df["protein_YFP_degtagged"][first_index:last_index]
        cell_count[start:stop] += 1

plt.subplot(312)      
plt.plot(timepoints, YFP, color = color_list[generation], alpha = .5)

plt.subplot(313)
plt.plot(timepoints, cell_count, color = color_list[generation], alpha = .5)

plt.show()