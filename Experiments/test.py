import numpy as np

time_lst =np.load("Simulations/sim_TU_data/time_culture.npy")
X_lst = np.load("Simulations/sim_TU_data/yfp_culture.npy")
Y_lst = np.load("Simulations/sim_TU_data/param_labels_culture.npy")

ktl = Y_lst[:, 0]
kdil = Y_lst[:, 1]
mrna = Y_lst[:, 2]
yfp_final = X_lst[:, -1]
yfp_penult = X_lst[:, -2]

dt = time_lst[-1] - time_lst[-2]

# 0 = (ktl * M) - (kdil * A) - dAdt
eps = 1e-6
res = ktl * mrna - kdil * yfp_final - ((yfp_final - yfp_penult) / dt)
scale1 = abs(ktl * mrna) + abs(kdil * yfp_final) + abs((yfp_final - yfp_penult) / dt) + eps
loss_phys = (abs(res) / scale1)


# print("mrna (first 5): ", mrna[:5])
# print("ktl (first 5): ", ktl[:5])
# print("kdil (first 5): ", kdil[:5])
# print("yfp_final (first 5): ", yfp_final[:5])
# print("yfp_penult (first 5): ", yfp_penult[:5])
# print("res (res, first 5): ", res[:5])
print("loss_phys (first 5): ", loss_phys.max())