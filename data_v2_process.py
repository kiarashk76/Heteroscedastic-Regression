import numpy as np
import os
import pickle

num_agent = 6

params = {
    # experiment configs
    'num_runs': 1,
    'num_epochs': 800,
    'num_data_points': 2000,
    'plt_show': False,
    'plt_save': False,
    'plot_show_epoch_freq': 100,

    # agents configs
    'num_agents': num_agent,
    'names': ['het'] * num_agent,
    'hidden_layers_mu': [[]] * num_agent,
    'hidden_layers_var': [[]] * num_agent,
    'data_dim': 10,
    'hidden_layers_error': [[64, 64]] * 10,
    'batch_sizes': [128] * num_agent,
    'step_sizes': [2 ** -i for i in range(5, 16, 2)],
    'plot_colors': ['b'] * num_agent,
    'loss_type': ['2'] * num_agent,
    'bias_available': [True] * num_agent,
    'mu_training': [True] * 10,
}
params['hidden_layers_var'] = [[64, 64]] * 10



def load_all(params):
    root = "data_v2/"
    dirs_name = os.listdir(root)
    print(dirs_name)
    # for dir_n in dirs_name:
    #     dir_path = root + dir_n + "/"
    #     if not os.path.isdir(dir_path):
    #         continue
    #     print("appending " + dir_path)
    #     final_path = "data_v2/final_results"
    #     if dir_path == final_path+"/":
    #         continue
    #     if not os.path.exists(final_path):
    #         os.mkdir(final_path)
    #     if os.path.exists(final_path + "/" + dir_n + "_totall.p"):
    #         os.remove(final_path + "/"+ dir_n + "_totall.p")
    #         print("file deleted")
    #
    #     files_name = os.listdir(dir_path)
    #     list_files = []
    #
    #     # print(files_name.__len__())
    #
    #
    #     for file_n in files_name:
    #         if file_n.endswith(".p"):
    #             file_path = dir_path + file_n
    #             dict = np.load(file_path, allow_pickle=True)
    #
    #         list_files.append(dict)
    #     # print(list_files[0])
    #     learn_mu = list_files[0]['last_mu']
    #     last_var = list_files[0]['last_var']
    #     error_list = list_files[0]['mu_error_list']
    #     error_list_sigma = list_files[0]['sigma_error_list']
    #     parameters = list_files[0]['params']
    #     # print(learn_mu.shape)
    #     try:
    #         w = list_files[0]['w']
    #     except:
    #         pass
    #
    #     for d in list_files[1:]:
    #         learn_mu = np.concatenate((learn_mu, d['last_mu']))
    #         last_var = np.concatenate((last_var, d['last_var']))
    #         error_list = np.concatenate((error_list, d['mu_error_list']))
    #         error_list_sigma = np.concatenate((error_list_sigma, d['sigma_error_list']))
    #
    #
    #
    #     with open(final_path+"/" + dir_n + "_totall.p", 'wb') as f:
    #         data = {
    #             'x': list_files[0]['x'],
    #             'y': list_files[0]['y'],
    #             'last_mu': learn_mu,
    #             'last_var': last_var,
    #             'mu_error_list': error_list,
    #             'sigma_error_list': error_list_sigma,
    #             'params': parameters
    #         }
    #         pickle.dump(data, f)
    #         # print(data)

if __name__ == "__main__":
    load_all(params)




