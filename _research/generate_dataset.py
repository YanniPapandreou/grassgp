import jax.numpy as np
from jax import vmap

from grassgp.grassmann import grass_log, valid_grass_point, compute_barycenter
from grassgp.utils import get_save_path 
from grassgp.utils import safe_save_jax_array_dict as safe_save

# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (10,6)

# generate dataset
N = 40
s_test = np.linspace(0, 1, N)
k = 2 * np.pi
x = np.cos(k * s_test).reshape(-1, 1)
y = np.sin(k * s_test).reshape(-1, 1)
Ws_test = np.hstack((x,y))[:,:,None]
assert vmap(valid_grass_point)(Ws_test).all()
d, n = Ws_test.shape[1:]

# # plot dataset
# for i in range(d):
#     plt.plot(s_test, Ws_test[:,i,0])
#     plt.title(f'{i+1}th component of projection')
#     plt.grid()
#     plt.xlabel(r'$s$')
#     plt.show()

# subsample data
s_gap = 4
s_train = s_test[::s_gap].copy()
Ws_train = Ws_test[::s_gap,:,:].copy()

# compute barycenter of train data
anchor_point = np.array(compute_barycenter(Ws_train))
assert valid_grass_point(anchor_point)
# print(f"anchor_point = {anchor_point.tolist()}")

# compute log of training data and full data
log_Ws_train = vmap(lambda W: grass_log(anchor_point, W))(Ws_train)
log_Ws_test = vmap(lambda W: grass_log(anchor_point, W))(Ws_test)

# for i in range(d):
#     plt.plot(s_test, Ws_test[:,i,0])
#     plt.scatter(s_train, Ws_train[:,i,0], c='r')
#     plt.title(f'{i+1}th component of projection')
#     plt.grid()
#     plt.xlabel(r'$s$')
#     plt.show()

# save training and testing data
training_test_data = {'s_train': s_train, 'Ws_train': Ws_train, 's_test': s_test, 'Ws_test': Ws_test, 'log_Ws_train': log_Ws_train, 'log_Ws_test': log_Ws_test, 'anchor_point': anchor_point}
head = './datasets'
main_name_training_test = "training_test_data_gpsr_example"
path_training_test = get_save_path(head, main_name_training_test)
print(path_training_test)
try:
    safe_save(path_training_test, training_test_data)
except FileExistsError:
    print("File exists so not saving.")
