import pickle
import numpy as np
import os
from Exp_Bias import *
from Exp_IrreducibleError import *
import matplotlib.pyplot as plt
from Experiment import *
from ExpMultiDim_single_y import *
from ExpMultiDim import *
from tqdm import tqdm

legend_size = 6
exp_map_name = {
    'Irre_linearNoise':experiment_irreducible_error1,
    'RegularReg_Irre_linearNoise':experiment_irreducible_error1,
    'Irre_fixedMu_linearNoise': experiment_irreducible_error1,
    'IrreBias_fixedMu_linearNoise_fixedBias': experiment_irreducible_error2,
    'Bias_fixedMu_linearBias': experiment_bias1,
    'Irre_fixedMu_rangeUniformNoise': experiment_irreducible_error4,
    'Bias_fixedMu_rangefixedBias': experiment_bias2,
    'Bias_fixedMu_rangelinearBias': experiment_bias4,
    'IrreBias_fixedMu_sinNoise_fixedBias': experiment_irreducible_error3,
    'Bias_rangefixedBias': experiment_bias2,
    'RegularReg_Bias_rangefixedBias': experiment_bias2,
    'Bias_rangelinearBias': experiment_bias4,
    'RegularReg_Bias_rangelinearBias': experiment_bias4,
    'Irre_rangeUniformNoise': experiment_irreducible_error4,
    'RegularReg_Irre_rangeUniformNoise': experiment_irreducible_error4,
    'Bias_quadraticBias1': experiment_bias3,
    'Bias_quadraticBias2': experiment_bias3,
    'Bias_quadraticBias3': experiment_bias3,
    'RegularReg_Bias_quadraticBias': experiment_bias3,
    'Bias_quadraticBias': experiment_bias3,
    'RegularReg_Bias_quadraticBias1': experiment_bias3,
    'RegularReg_Bias_quadraticBias2': experiment_bias3,
    'RegularReg_Bias_quadraticBias3': experiment_bias3,
    'Bias_FancyquadraticBias1': experiment_bias5,
    'Bias_FancyquadraticBias2': experiment_bias5,
    'Bias_FancyquadraticBias3': experiment_bias5,
    'RegularReg_Bias_FancyquadraticBias1': experiment_bias5,
    'RegularReg_Bias_FancyquadraticBias2': experiment_bias5,
    'RegularReg_Bias_FancyquadraticBias3': experiment_bias5,
    'HetFail1_Irre_rangeUniformNoise1': experiment_irreducible_error5,
    'HetFail1_Irre_rangeUniformNoise2': experiment_irreducible_error5,
    'HetFail1_Irre_rangeUniformNoise3': experiment_irreducible_error5,
    'HetFail1_Irre_rangeUniformNoise4': experiment_irreducible_error5,
    'HetFail1_Irre_rangeUniformNoise5': experiment_irreducible_error5,
    'RegFail1_Irre_rangeUniformNoise1': experiment_irreducible_error5,
    'RegFail1_Irre_rangeUniformNoise2': experiment_irreducible_error5,
    'RegFail1_Irre_rangeUniformNoise3': experiment_irreducible_error5,
    'RegFail1_Irre_rangeUniformNoise4': experiment_irreducible_error5,
    'RegFail1_Irre_rangeUniformNoise5': experiment_irreducible_error5,
    'HetFail1_Irre_rangeUniformNoise1AAA': experiment_irreducible_error10,
    'HetFail1_Irre_rangeUniformNoise2AAA': experiment_irreducible_error10,
    'HetFail1_Irre_rangeUniformNoise3AAA': experiment_irreducible_error10,
    'HetFail1_Irre_rangeUniformNoise4AAA': experiment_irreducible_error10,
    'HetFail1_Irre_rangeUniformNoise5AAA': experiment_irreducible_error10,
    'RegFail1_Irre_rangeUniformNoise1AAA': experiment_irreducible_error10,
    'RegFail1_Irre_rangeUniformNoise2AAA': experiment_irreducible_error10,
    'RegFail1_Irre_rangeUniformNoise3AAA': experiment_irreducible_error10,
    'RegFail1_Irre_rangeUniformNoise4AAA': experiment_irreducible_error10,
    'RegFail1_Irre_rangeUniformNoise5AAA': experiment_irreducible_error10,
    'HetFail2_Irre_rangeUniformNoise1AAA': experiment_irreducible_error10,
    'HetFail2_Irre_rangeUniformNoise2AAA': experiment_irreducible_error10,
    'HetFail2_Irre_rangeUniformNoise3AAA': experiment_irreducible_error10,
    'HetFail2_Irre_rangeUniformNoise4AAA': experiment_irreducible_error10,
    'HetFail2_Irre_rangeUniformNoise15AAA': experiment_irreducible_error10,
    'MD_Bias_quadraticBias1_MSE_totall.p': MD_experiment_quadraticBias,
    'MD_Bias_quadraticBias1_totall.p': MD_experiment_quadraticBias,
    'MD_Bias_rangelinearBias_MSE_totall.p': MD_experiment_rangeLBias,
    'MD_Bias_rangelinearBias_totall.p': MD_experiment_rangeLBias,
    'MD_experiment_irreducible_error_single_y_MSE_totall.p': MD_experiment_irreducible_error_single_y,
    'MD_experiment_irreducible_error_single_y_totall.p': MD_experiment_irreducible_error_single_y,
    'MD_experiment_irreducible_linear_single_y_MSE_totall.p': MD_experiment_irreducible_linear_single_y,
    'MD_experiment_irreducible_linear_single_y_totall.p': MD_experiment_irreducible_linear_single_y,
    'MD_experiment_quadraticBias_single_y_MSE_totall.p': MD_experiment_quadraticBias_single_y,
    'MD_experiment_quadraticBias_single_y_totall.p': MD_experiment_quadraticBias_single_y,
    'MD_experiment_rangeLBias_single_y_MSE_totall.p': MD_experiment_rangeLBias_single_y,
    'MD_experiment_rangeLBias_single_y_totall.p': MD_experiment_rangeLBias_single_y,
    'MD_Irre_linearNoise_MSE_totall.p': MD_experiment_irreducible_linear,
    'MD_Irre_linearNoise_totall.p': MD_experiment_irreducible_linear,
    'MD_Irre_rangeUniformNoise_MSE_totall.p': MD_experiment_irreducible_error,
    'MD_Irre_rangeUniformNoise_totall.p': MD_experiment_irreducible_error,
}


def compute_mu_accuracy(mu, true_mu, threshold, type=0):
    diff = np.abs(mu - true_mu)
    if type == 0:
        acceptance = [diff<=threshold]
        p = np.mean(acceptance[0])
        performance =  np.sum(acceptance) / len(mu)
    if type == 1:
        sigmoid = 1 / (1 + np.exp(-threshold * diff))
        acceptance = 1 - (sigmoid - 0.5) * 2
        performance = np.sum(acceptance) / len(mu)

    return performance

def recreate_dataset(exp_name, params):
    exp = exp_map_name[exp_name](params, exp_name)
    np.random.seed(0)
    exp.create_dataset()
    return exp.x, exp.y, exp.mu

def drawPlotUncertainty(x, y, y_err, label, color, axis):
    if color == "blue":
        axis.plot(x, y, color, label=label)
        # axis.fill_between(x,
        #             y - y_err,
        #             y + y_err,
        #             facecolor='lightcyan', alpha=.7, edgecolor='none')
        axis.fill_between(x,
                          y - y_err,
                          y + y_err,
                          facecolor='blue', alpha=.2, edgecolor='none')
    elif color == "red":
        axis.plot(x, y, color, label=label)
        # axis.fill_between(x,
        #             y - y_err,
        #             y + y_err,
        #             facecolor='bisque',alpha=.7,  edgecolor='none')
        axis.fill_between(x,
                          y - y_err,
                          y + y_err,
                          facecolor='red', alpha=.2, edgecolor='none')
    else:
        raise NotImplemented("not implemented")

def best_stepsize_wrt_thresh(learn_mu, true_mu, step_sizes, threshold, name, type):
    max_value = -np.inf
    max_index = -1
    p = []
    print("finding the best step size for "+name+"... \n with threshold="+ str(threshold))
    for s_index in step_sizes:
    # for s_index in range(learn_mu.shape[1]):
        performance = []
        for e in range(learn_mu.shape[0]):
            accuracy = compute_mu_accuracy(learn_mu[e][s_index], true_mu, threshold, type)
            performance.append(accuracy)
        value = sum(performance[-100:])
        if value >= max_value:
            p = performance
            max_value = value
            max_index = s_index
    return max_index, p

def best_stepsize_wrt_thresh2(learn_mu, true_mu, step_sizes, threshold, name, type):
    def compute_mu_accuracy2(mu, true_mu, threshold, type=0):
        diff = np.abs(mu - true_mu)
        if type == 0:
            acceptance = [diff[:] <= threshold]
            # performance = np.sum(acceptance[0]) / len(mu)
            performance = np.mean(acceptance[0], axis=1)
        if type == 1:
            sigmoid = 1 / (1 + np.exp(-threshold * diff[:]))
            acceptance = 1 - (sigmoid - 0.5) * 2
            # performance = np.sum(acceptance) / len(mu)
            performance = np.mean(acceptance, axis=1)

        return performance

    max_value = -np.inf
    max_index = -1
    max_std = 0
    # print("finding the best step size for "+name+"... \n with threshold="+ str(threshold))
    for s_index in step_sizes:
        accuracy = compute_mu_accuracy2(learn_mu[:, -1, s_index], true_mu, threshold, type)
        performance = np.mean(accuracy, axis=0).mean()
        std = np.std(accuracy, axis=0).mean()
        if performance > max_value:
            max_value = performance
            max_std = std
            max_index = s_index
    return max_index, max_value, max_std

def best_stepsize_wrt_mse(mse_list, name):
    max_value = np.inf
    max_index = -1
    # print("finding the best step size for "+name+"... with mse")
    avg_mse = np.mean(mse_list, axis=0)
    step_size_index_list = list(range(avg_mse.shape[1]))
    for s_index in range(mse_list.shape[2]):
        value = np.sum(avg_mse[-100:, s_index])
        value1 = np.mean(avg_mse[-100:,s_index])
        value2 = np.mean(avg_mse[-10:, s_index])
        # print(np.abs(value1 - value2))
        # if np.abs(value1 - value2) > 0.5:
        #     step_size_index_list.remove(s_index)
        #     continue
        if value < max_value:
            max_value = value
            max_index = s_index
    return max_index, step_size_index_list

def best_stepsize_wrt_var(var_mse_list, name):
    max_value = np.inf
    max_index = -1
    print("finding the best step size for "+name+"... with mse")
    avg_mse = np.mean(var_mse_list, axis=0)
    step_size_index_list = list(range(avg_mse.shape[1]))
    for s_index in range(var_mse_list.shape[2]):
        value = np.sum(avg_mse[-5:, s_index])
        # value1 = np.mean(avg_mse[-100:,s_index])
        # value2 = np.mean(avg_mse[-10:, s_index])
        # print(np.abs(value1 - value2))
        # if np.abs(value1 - value2) > 0.5:
        #     step_size_index_list.remove(s_index)
        #     continue
        if value < max_value:
            max_value = value
            max_index = s_index
    return max_index, step_size_index_list

def draw_mse_stepsize(data, axs, color, label, step_size_index, title=True):
    #mse plot
    avg_mu_error = np.mean(data['mu_error_list'], axis=0)
    std_mu_error = 1/2 *np.std(data['mu_error_list'], axis=0)

    # axs.plot(avg_mu_error[:, step_size_index], color=color, label=label)
    drawPlotUncertainty(range(len(avg_mu_error[:, step_size_index])),
                        avg_mu_error[:, step_size_index], std_mu_error[:, step_size_index], label, color, axs)
    step_size = np.log2(data['params']['step_sizes'][step_size_index])
    if title:
        axs.title.set_text("step_size: 2^" + str(step_size))
    axs.legend(prop={'size': legend_size})
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

def draw_var_mse_stepsize(data, axs, color, label, step_size_index, title=True):
    #mse plot
    avg_var_error = np.mean(data['sigma_error_list'], axis=0)
    std_var_error = 1/2 *np.std(data['sigma_error_list'], axis=0)

    # axs.plot(avg_var_error[:, step_size_index], color=color, label=label)
    drawPlotUncertainty(range(len(avg_var_error[:, step_size_index])),
                        avg_var_error[:, step_size_index], std_var_error[:, step_size_index], label, color, axs)
    step_size = np.log2(data['params']['step_sizes'][step_size_index])
    if title:
        axs.title.set_text("step_size: 2^" + str(step_size))
    axs.legend(prop={'size': legend_size})
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

def draw_learned_mu(data, axs, color, label, step_size_index, epoch_number, x, true_mu, het=True, ground_truth=True):
    mu = data['avg_learn_mu'][epoch_number,step_size_index]
    var = data['avg_learn_var'][epoch_number,step_size_index]
    if not het:
        var = np.zeros_like(mu)
    drawPlotUncertainty(x[:,0], mu[:,0], var[:,0], label=label, color=color, axis=axs)
    step_size = np.log2(data['params']['step_sizes'][step_size_index])
    axs.title.set_text("step_size: 2^"+str(step_size))
    if ground_truth:
        axs.plot(x[:,0], true_mu[:,0], 'ko', markersize=0.5, label='GT', alpha=0.5)
    axs.legend(prop={'size': legend_size})
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

def main1(threshold_hard, threshold_soft, path):
    results = os.listdir(path)
    fig1, axs_mse = plt.subplots(2, 3, constrained_layout=False)
    fig4, axs_var_mse = plt.subplots(2, 3, constrained_layout=False)

    fig2, axs_mu = plt.subplots(2, 3, constrained_layout=True)
    gt = np.ones_like(axs_mu)

    fig1.text(.5, .001, "Mean-Square-Error for Learned Mu", ha='center')
    fig4.text(.5, .001, "Mean-Square-Error for Learned Variance", ha='center')
    fig2.text(.5, .001, "Learned-Function", ha='center')

    fig3, axs_best_hard = plt.subplots(2, 2, constrained_layout=True)
    fig3.text(.5, .001, "best parameter plot hard thresh", ha='center')

    fig5, axs_best_soft = plt.subplots(2, 2, constrained_layout=True)
    fig5.text(.5, .001, "best parameter plot soft thresh", ha='center')

    for file_name in results:
        if 'Reg' in file_name:
            label = "Reg"
            color = "red"
        else:
            label = "Het"
            color = "blue"
        with open(path+"/"+file_name, 'rb') as fp:
            if file_name == ".DS_Store":
                continue
            data = pickle.load(fp)
        exp_name = file_name.split('.')[0]
        x, y, true_mu = recreate_dataset(exp_name, data['params'])
        learn_mu = data['avg_learn_mu']

        #mu mse plot
        i, j = 0, 0
        for s_index in range(data['mu_error_list'].shape[2]):
            draw_mse_stepsize(data, axs_mse[i, j], color, label, s_index)
            step_size = np.log2(data['params']['step_sizes'][s_index])
            axs_mse[i][j].title.set_text("step_size: 2^" + str(step_size))
            i += 1
            if i == 2:
                i = 0
                j += 1

        #var mse plot
        i, j = 0, 0
        for s_index in range(data['sigma_error_list'].shape[2]):
            draw_var_mse_stepsize(data, axs_var_mse[i, j], color, label, s_index)
            step_size = np.log2(data['params']['step_sizes'][s_index])
            axs_var_mse[i][j].title.set_text("step_size: 2^" + str(step_size))
            i += 1
            if i == 2:
                i = 0
                j += 1

        #learned mu plot
        i, j = 0, 0
        for s_index in range(learn_mu.shape[1]):
            if label=="Het":
                het=True
            else:
                het=False
            draw_learned_mu(data, axs_mu[i,j], color, label, s_index, -1, x, y, het=het, ground_truth=gt[i,j])
            step_size = np.log2(data['params']['step_sizes'][s_index])
            axs_mu[i][j].title.set_text("step_size: 2^"+str(step_size))
            gt[i,j] = 0
            i += 1
            if i == 2:
                i = 0
                j += 1


        #best parameters plot
        best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data['mu_error_list'], exp_name)
        print(converged_step_sizes)
        step_size = np.log2(data['params']['step_sizes'][best_s_mse])
        axs_best_hard[0,0].set_title("mse")
        draw_mse_stepsize(data, axs_best_hard[0, 0], color, label + "2^" + str(step_size), best_s_mse, title=False) #mse
        i, j = 0, 1
        for t in threshold_hard:
            # hard-threshold plot
            best_s_thresh, performance = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, t, exp_name,
                                                                  type=0)
            if True:
                step_size = np.log2(data['params']['step_sizes'][best_s_thresh])
                axs_best_hard[i,j].plot(performance, label=label+"2^"+str(step_size), color=color)
                axs_best_hard[i,j].set_title('threshold='+str(t))
                axs_best_hard[i,j].legend(prop={'size': legend_size})
                axs_best_hard[i,j].spines['top'].set_visible(False)
                axs_best_hard[i,j].spines['right'].set_visible(False)
                axs_best_hard[i,j].set_ylim(0,1)
            j += 1
            if j == 2:
                j = 0
                i += 1

        step_size = np.log2(data['params']['step_sizes'][best_s_mse])
        axs_best_soft[0, 0].set_title("mse")
        draw_mse_stepsize(data, axs_best_soft[0, 0], color, label + "2^" + str(step_size), best_s_mse,
                          title=False)  # mse
        i, j = 0, 1
        for t in threshold_soft:
            # soft-threshold plot
            best_s_thresh, performance = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, t, exp_name,
                                                                  type=1)
            if True:
                step_size = np.log2(data['params']['step_sizes'][best_s_thresh])
                axs_best_soft[i, j].plot(performance, label=label + "2^" + str(step_size), color=color)
                axs_best_soft[i, j].set_title('sigmoid par=' + str(t))
                axs_best_soft[i, j].legend(prop={'size': legend_size})
                axs_best_soft[i, j].spines['top'].set_visible(False)
                axs_best_soft[i, j].spines['right'].set_visible(False)
                axs_best_soft[i, j].set_ylim(0, 1)
            j += 1
            if j == 2:
                j = 0
                i += 1
    # fig1.show()
    fig2.show()
    # fig3.show()
    fig4.show()
    # fig5.show()
    # fig1.savefig("data/Plots/"+path.split("/")[1]+str("_MuMSE.svg"), format='svg')
    # fig4.savefig("data/Plots/"+path.split("/")[1]+str("_VarMSE.svg"), format='svg')
    # fig2.savefig("data/Plots/"+path.split("/")[1]+str("_LearnedMu.svg"), format='svg')
    # fig3.savefig("data/Plots/"+path.split("/")[1]+str("_BestParHard.svg"), format='svg')
    # fig5.savefig("data/Plots/"+path.split("/")[1]+str("_BestParSoft.svg"), format='svg')

def main2(path):
    results = os.listdir(path)
    fig_hard, ax_hard = plt.subplots(1, 1, constrained_layout=False)
    fig_soft, ax_soft = plt.subplots(1, 1, constrained_layout=False)
    for file_name in results:
        if 'Reg' in file_name:
            label = "Reg"
            color = "red"
        else:
            label = "Het"
            color = "blue"
        with open(path + "/" + file_name, 'rb') as fp:
            if file_name == ".DS_Store":
                continue
            data = pickle.load(fp)
        exp_name = file_name.split('.')[0]
        x, y, true_mu = recreate_dataset(exp_name, data['params'])
        # learn_mu = data['avg_learn_mu']
        learn_mu = data['last_mu']
        #best parameters plot hard threshold
        best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data['mu_error_list'], exp_name)
        threshold = np.arange(0.0, 0.001, 0.00001)
        threshold = np.concatenate((threshold, np.arange(0.001, 0.1, 0.01)))
        threshold = np.concatenate((threshold, np.arange(0.1, 1.0, 0.1)))
        avg_performance_list = []
        std_performance_list = []
        for t in threshold:
            best_s_thresh, performance = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, t, exp_name,
                                                                  type=0)
            avg_performance = np.mean(performance[-100:])
            std_performance = np.std(performance[-100:])
            avg_performance_list.append(avg_performance)
            std_performance_list.append(std_performance)
        # ax_hard.plot(threshold, avg_performance_list, label=label, color=color)
        ax_hard.set_xscale("log")
        drawPlotUncertainty(threshold, np.asarray(avg_performance_list), np.asarray(std_performance_list)
                            , label=label, color=color, axis=ax_hard)

        #best parameters plot soft threshold
        # best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data['mu_error_list'], exp_name)
        # threshold = np.arange(0, 64, 4)
        # avg_performance_list = []
        # for t in threshold:
        #     best_s_thresh, performance = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, t, exp_name,
        #                                                           type=1)
        #     avg_performance = np.mean(performance[-100:])
        #     avg_performance_list.append(avg_performance)
        # ax_soft.plot(threshold, avg_performance_list, label=label, color=color)

    ax_hard.set_title(path)
    ax_hard.set_xlabel("hard threshold")
    ax_hard.set_ylabel("performance last epoch best par")
    ax_hard.legend(prop={'size': legend_size})
    # fig_hard.show()
    ax_hard.spines['top'].set_visible(False)
    ax_hard.spines['right'].set_visible(False)
    # fig_hard.savefig("data/Plots/"+path.split("/")[1]+str("hard threshold comparison.eps"), format='eps')
    fig_hard.show()
    ax_soft.set_title(path)
    ax_soft.set_xlabel("sigmoid parameter")
    ax_soft.set_ylabel("performance last epoch best par")
    ax_soft.legend(prop={'size': legend_size})
    # fig_soft.show()
    ax_soft.spines['top'].set_visible(False)
    ax_soft.spines['right'].set_visible(False)
    # fig_soft.savefig("data/Plots/" + path.split("/")[1] + str("soft threshold comparison.eps"), format='eps')
    fig_soft.show()

def main3(hard_threshold, soft_threshold):
    path = "data/HetFail1_Irre_rangeUniformNoise"
    figs_mu, axs_mu = [], []
    figs_comp_hard, axs_comp_hard = [], []
    figs_comp_soft, axs_comp_soft = [], []
    for i in range(5):
        fig, ax = plt.subplots(1, 1, constrained_layout=False)
        figs_mu.append(fig)
        axs_mu.append(ax)
        fig, ax = plt.subplots(1, 1, constrained_layout=False)
        figs_comp_hard.append(fig)
        axs_comp_hard.append(ax)
        fig, ax = plt.subplots(1, 1, constrained_layout=False)
        figs_comp_soft.append(fig)
        axs_comp_soft.append(ax)

    gt = np.ones_like(axs_mu)
    results = os.listdir(path)

    mu_capacity_performance_hard = np.zeros([5, 2, len(hard_threshold)])
    mu_capacity_performance_soft = np.zeros([5, 2, len(soft_threshold)])
    for file_name in results:
        if 'Reg' in file_name:
            label = "Reg"
            color = "red"
        else:
            label = "Het"
            color = "blue"
        with open(path+"/"+file_name, 'rb') as fp:
            if file_name == ".DS_Store":
                continue
            data = pickle.load(fp)
        exp_name = file_name.split('.')[0]
        x, y, true_mu = recreate_dataset(exp_name, data['params'])
        learn_mu = data['avg_learn_mu']

        #learn mu plot
        best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data['mu_error_list'], exp_name)
        print(best_s_mse, file_name)
        index = int(file_name[-3])-1
        draw_learned_mu(data, axs_mu[index], color, label, best_s_mse, -1, x, y, het=label=="Het", ground_truth=gt[index])
        gt[index] = 0
        axs_mu[index].title.set_text(data['params']['hidden_layers_mu'][0])

        #performance for different mu capacity hard threshold
        for counter, t in enumerate(hard_threshold):
            best_s_thresh, performance = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, t, exp_name, type=0)
            avg_performance = np.mean(performance[-100:])
            if data['params']['hidden_layers_mu'][0] == [8, 8]:
                if label=="Het":
                    mu_capacity_performance_hard[0, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_hard[0, 1, counter] = avg_performance

            if data['params']['hidden_layers_mu'][0] == [12, 12]:
                if label=="Het":
                    mu_capacity_performance_hard[1, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_hard[1, 1, counter] = avg_performance

            if data['params']['hidden_layers_mu'][0] == [16, 16]:
                if label=="Het":
                    mu_capacity_performance_hard[2, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_hard[2, 1, counter] = avg_performance

            if data['params']['hidden_layers_mu'][0] == [24, 24]:
                if label=="Het":
                    mu_capacity_performance_hard[3, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_hard[3, 1, counter] = avg_performance
            if file_name == ".DS_Store":
                continue
            if data['params']['hidden_layers_mu'][0] == [32, 32]:
                if label=="Het":
                    mu_capacity_performance_hard[4, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_hard[4, 1, counter] = avg_performance

        #performance for different mu capacity soft threshold
        for counter, t in enumerate(soft_threshold):
            best_s_thresh, performance = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, t, exp_name, type=1)
            avg_performance = np.mean(performance[-100:])
            if data['params']['hidden_layers_mu'][0] == [8, 8]:
                if label=="Het":
                    mu_capacity_performance_soft[0, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_soft[0, 1, counter] = avg_performance

            if data['params']['hidden_layers_mu'][0] == [12, 12]:
                if label=="Het":
                    mu_capacity_performance_soft[1, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_soft[1, 1, counter] = avg_performance

            if data['params']['hidden_layers_mu'][0] == [16, 16]:
                if label=="Het":
                    mu_capacity_performance_soft[2, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_soft[2, 1, counter] = avg_performance

            if data['params']['hidden_layers_mu'][0] == [24, 24]:
                if label=="Het":
                    mu_capacity_performance_soft[3, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_soft[3, 1, counter] = avg_performance
            if file_name == ".DS_Store":
                continue
            if data['params']['hidden_layers_mu'][0] == [32, 32]:
                if label=="Het":
                    mu_capacity_performance_soft[4, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_soft[4, 1, counter] = avg_performance

    plt.setp(axs_comp_hard, xticks=[0, 1, 2, 3, 4], xticklabels=['[8,8]', '[12,12]', '[16,16]', '[24,24]', '[32,32]'],)
    for counter, t in enumerate(hard_threshold):
        axs_comp_hard[counter].plot(mu_capacity_performance_hard[:, 0, counter], label = "Het", color = "blue")
        axs_comp_hard[counter].plot(mu_capacity_performance_hard[:, 1, counter], label = "Reg", color = "red")
        axs_comp_hard[counter].title.set_text("Hard_threshold:"+ str(t))
        axs_comp_hard[counter].spines['top'].set_visible(False)
        axs_comp_hard[counter].spines['right'].set_visible(False)
        axs_comp_hard[counter].legend()
        # figs_comp_hard[counter].show()
        figs_comp_hard[counter].savefig("data/Plots/FailExample, threshold:"+str(t)+".eps", format='eps')

    plt.setp(axs_comp_soft, xticks=[0, 1, 2, 3, 4], xticklabels=['[8,8]', '[12,12]', '[16,16]', '[24,24]', '[32,32]'],)
    for counter, t in enumerate(soft_threshold):
        axs_comp_soft[counter].plot(mu_capacity_performance_soft[:, 0, counter], label = "Het", color = "blue")
        axs_comp_soft[counter].plot(mu_capacity_performance_soft[:, 1, counter], label = "Reg", color = "red")
        axs_comp_soft[counter].title.set_text("Soft_threshold:"+ str(t))
        axs_comp_soft[counter].spines['top'].set_visible(False)
        axs_comp_soft[counter].spines['right'].set_visible(False)
        axs_comp_soft[counter].legend()
        # figs_comp_soft[counter].show()
        figs_comp_soft[counter].savefig("data/Plots/FailExample, sigmoid par:"+str(t)+".eps", format='eps')

    for counter, fig in enumerate(figs_mu):
        fig.savefig("data/Plots/FailExampleMu"+str(counter)+".eps", format='eps')
        # fig.show()

def main(path):
    results = os.listdir(path)
    fig_hard, ax_hard = plt.subplots(1, 1, constrained_layout=False)
    fig_soft, ax_soft = plt.subplots(1, 1, constrained_layout=False)
    for file_name in results:
        if 'Reg' in file_name:
            label = "Reg"
            color = "red"
        else:
            label = "Het"
            color = "blue"
        with open(path + "/" + file_name, 'rb') as fp:
            if file_name == ".DS_Store":
                continue
            data = pickle.load(fp)
        exp_name = file_name.split('.')[0]
        x, y, true_mu = recreate_dataset(exp_name, data['params'])
        # x, y, true_mu
        # learn_mu = data['avg_learn_mu']
        learn_mu = data['last_mu']
        #best parameters plot hard threshold
        best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data['mu_error_list'], exp_name)
        threshold = np.arange(0.0, 0.001, 0.00001)
        threshold = np.concatenate((threshold, np.arange(0.001, 0.1, 0.01)))
        threshold = np.concatenate((threshold, np.arange(0.1, 1.0, 0.1)))
        avg_performance_list = []
        std_performance_list = []
        for t in threshold:
            best_s_thresh, avg_performance, std_performance= best_stepsize_wrt_thresh2(learn_mu, true_mu, converged_step_sizes, t, exp_name,
                                                                  type=0)

            avg_performance_list.append(avg_performance)
            std_performance_list.append(std_performance)
        ax_hard.set_xscale("log")
        drawPlotUncertainty(threshold, np.asarray(avg_performance_list)[:, 0], np.asarray(std_performance_list)[:, 0]
                            , label=label, color=color, axis=ax_hard)

        #best parameters plot soft threshold
        best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data['mu_error_list'], exp_name)
        threshold = np.arange(0, 128, 1)
        avg_performance_list = []
        std_performance_list = []
        for t in threshold:
            best_s_thresh, avg_performance, std_performance = best_stepsize_wrt_thresh2(learn_mu, true_mu, converged_step_sizes, t, exp_name,
                                                                  type=1)
            avg_performance_list.append(avg_performance)
            std_performance_list.append(std_performance)
        # ax_soft.plot(threshold, avg_performance_list, label=label, color=color)
        # ax_soft.set_xscale("log")
        drawPlotUncertainty(threshold, np.asarray(avg_performance_list)[:, 0], np.asarray(std_performance_list)[:, 0]
                            , label=label, color=color, axis=ax_soft)

    ax_hard.set_title(path)
    ax_hard.set_xlabel("hard-threshold parameter (\u03B7)")
    ax_hard.set_ylabel("performance last epoch best par")
    ax_hard.legend(prop={'size': legend_size})
    ax_hard.spines['top'].set_visible(False)
    ax_hard.spines['right'].set_visible(False)
    # fig_hard.savefig("data/Plots/1/"+path.split("/")[1]+str("hard threshold comparison.svg"), format='svg')
    fig_hard.show()

    ax_soft.set_title(path)
    ax_soft.set_xlabel("soft-threshold parameter (\u03B7)")
    ax_soft.set_ylabel("performance last epoch best par")
    ax_soft.legend(prop={'size': legend_size})
    ax_soft.spines['top'].set_visible(False)
    ax_soft.spines['right'].set_visible(False)
    # fig_soft.savefig("data/Plots/1/" + path.split("/")[1] + str("soft threshold comparison.svg"), format='svg')
    fig_soft.show()

def main4(path):
    results = os.listdir(path)
    fig1, axs_mse = plt.subplots(1, 1, constrained_layout=False)
    fig4, axs_var_mse = plt.subplots(1, 1, constrained_layout=False)

    fig2, axs_mu = plt.subplots(1, 1, constrained_layout=True)

    fig1.text(.5, .001, "Mean-Square-Error for Learned Mu", ha='center')
    fig4.text(.5, .001, "Mean-Square-Error for Learned Variance", ha='center')
    fig2.text(.5, .001, "Learned-Function", ha='center')

    fig3, axs_best_hard = plt.subplots(2, 2, constrained_layout=True)
    fig3.text(.5, .001, "best parameter plot hard thresh", ha='center')

    fig5, axs_best_soft = plt.subplots(2, 2, constrained_layout=True)
    fig5.text(.5, .001, "best parameter plot soft thresh", ha='center')
    gt = True
    for i, file_name in enumerate(results):
        if 'Reg' in file_name:
            label = "Reg"
            color = "red"
        else:
            label = "Het"
            color = "blue"
        with open(path+"/"+file_name, 'rb') as fp:
            if file_name == ".DS_Store":
                continue
            data = pickle.load(fp)
        exp_name = file_name.split('.')[0]
        x, y, true_mu = recreate_dataset(exp_name, data['params'])
        learn_mu = data['avg_learn_mu']

        #mu mse plot
        best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data['mu_error_list'], exp_name)
        step_size = np.log2(data['params']['step_sizes'][best_s_mse])
        draw_mse_stepsize(data, axs_mse, color, label+ str(step_size), best_s_mse, title=False)
        axs_var_mse.set_yscale("log")
        draw_var_mse_stepsize(data, axs_var_mse, color, label+str(step_size), best_s_mse, title=False)

        step_size, _ = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, 0.5, "a", type=1)
        if "rangeUniform" in file_name:
            step_size = 3
        if "rangelinearBias" in file_name:
            step_size = 3
        draw_learned_mu(data, axs_mu, color, label+str(step_size), step_size, -1, x, y, het=label=="Het",
                            ground_truth=gt)
        gt = False
    fig2.show()
    fig2.savefig("data/Plots/1/" + path.split("/")[2] + str("mu.svg"), format='svg')
    # fig1.show()
    fig4.show()
    fig1.savefig("data/Plots/1/" + path.split("/")[2] + str("a.svg"), format='svg')
    fig4.savefig("data/Plots/1/" + path.split("/")[2] + str("b.svg"), format='svg')

def fixedMu(path):
    results = os.listdir(path)
    fig1, axs_mu = plt.subplots(2, 2, constrained_layout=False)
    fig4, axs_var_mse = plt.subplots(2, 2, constrained_layout=False)
    i, j = 0, 0
    for file_name in results:
        label = file_name
        color = "blue"
        with open(path+"/"+file_name, 'rb') as fp:
            if file_name == ".DS_Store":
                continue
            data = pickle.load(fp)
        exp_name = file_name.split('.')[0]
        x, y, true_mu = recreate_dataset(exp_name, data['params'])
        learn_mu = data['avg_learn_mu']

        #mu mse plot
        best_s_mse, converged_step_sizes = best_stepsize_wrt_var(data['sigma_error_list'], exp_name)
        if "rangeUniform" in file_name:
            best_s_mse = 3
        # best_s_mse = 3
        step_size = np.log2(data['params']['step_sizes'][best_s_mse])

        draw_var_mse_stepsize(data, axs_var_mse[i, j], color , label+str(step_size), best_s_mse, title=False)
        draw_learned_mu(data, axs_mu[i, j], color, label, best_s_mse, -1, x, y, het=True, ground_truth=True)
        i+=1
        if i == 2:
            i=0
            j+=1

    fig1.show()
    fig4.show()
    # fig1.savefig("data/Plots/1/" + path.split("/")[1] + str("mu.svg"), format='svg')
    # fig4.savefig("data/Plots/1/" + path.split("/")[1] + str("var.svg"), format='svg')

def hetfail1(hard_threshold, soft_threshold):
    path = "data/HetFail1"
    figs_mu, axs_mu = [], []
    figs_comp_hard, axs_comp_hard = [], []
    figs_comp_soft, axs_comp_soft = [], []
    for i in range(5):
        fig, ax = plt.subplots(1, 1, constrained_layout=False)
        figs_mu.append(fig)
        axs_mu.append(ax)
        fig, ax = plt.subplots(1, 1, constrained_layout=False)
        figs_comp_hard.append(fig)
        axs_comp_hard.append(ax)
        fig, ax = plt.subplots(1, 1, constrained_layout=False)
        figs_comp_soft.append(fig)
        axs_comp_soft.append(ax)

    gt = np.ones_like(axs_mu)
    results = os.listdir(path)

    mu_capacity_performance_hard = np.zeros([5, 2, len(hard_threshold)])
    var_capacity_performance_hard = np.zeros([5, 2, len(hard_threshold)])

    mu_capacity_performance_soft = np.zeros([5, 2, len(soft_threshold)])
    var_capacity_performance_soft = np.zeros([5, 2, len(soft_threshold)])
    for file_name in results:
        if 'Reg' in file_name:
            label = "Reg"
            color = "red"
        else:
            label = "Het"
            color = "blue"
        with open(path+"/"+file_name, 'rb') as fp:
            if file_name == ".DS_Store":
                continue
            data = pickle.load(fp)
        exp_name = file_name.split('.')[0]
        x, y, true_mu = recreate_dataset(exp_name, data['params'])
        learn_mu = data['avg_learn_mu']
        learn_mu = data['last_mu']
        #learn mu plot
        best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data['mu_error_list'], exp_name)
        print(best_s_mse, file_name)
        index = int(file_name[-6])-1
        draw_learned_mu(data, axs_mu[index], color, label, best_s_mse, -1, x, y, het=label=="Het", ground_truth=gt[index])
        gt[index] = 0
        axs_mu[index].title.set_text(data['params']['hidden_layers_mu'][0])

        #performance for different mu capacity hard threshold
        for counter, t in enumerate(hard_threshold):
            # best_s_thresh, performance = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, t, exp_name, type=0)
            best_s_thresh, avg_performance, avg_std = best_stepsize_wrt_thresh2(learn_mu, true_mu, converged_step_sizes, t, exp_name, type=0)
            # avg_performance = np.mean(performance[-100:])
            if data['params']['hidden_layers_mu'][0] == [8, 8]:
                if label=="Het":
                    mu_capacity_performance_hard[0, 0, counter] = avg_performance
                    var_capacity_performance_hard[0, 0, counter] = avg_std

                else:
                    mu_capacity_performance_hard[0, 1, counter] = avg_performance
                    var_capacity_performance_hard[0, 1, counter] = avg_std


            if data['params']['hidden_layers_mu'][0] == [12, 12]:
                if label=="Het":
                    mu_capacity_performance_hard[1, 0, counter] = avg_performance
                    var_capacity_performance_hard[1, 0, counter] = avg_std

                else:
                    mu_capacity_performance_hard[1, 1, counter] = avg_performance
                    var_capacity_performance_hard[1, 1, counter] = avg_std


            if data['params']['hidden_layers_mu'][0] == [16, 16]:
                if label=="Het":
                    mu_capacity_performance_hard[2, 0, counter] = avg_performance
                    var_capacity_performance_hard[2, 0, counter] = avg_std

                else:
                    mu_capacity_performance_hard[2, 1, counter] = avg_performance
                    var_capacity_performance_hard[2, 1, counter] = avg_std


            if data['params']['hidden_layers_mu'][0] == [24, 24]:
                if label=="Het":
                    mu_capacity_performance_hard[3, 0, counter] = avg_performance
                    var_capacity_performance_hard[3, 0, counter] = avg_std

                else:
                    mu_capacity_performance_hard[3, 1, counter] = avg_performance
                    var_capacity_performance_hard[3, 1, counter] = avg_std

            if file_name == ".DS_Store":
                continue
            if data['params']['hidden_layers_mu'][0] == [32, 32]:
                if label=="Het":
                    mu_capacity_performance_hard[4, 0, counter] = avg_performance
                    var_capacity_performance_hard[4, 0, counter] = avg_std

                else:
                    mu_capacity_performance_hard[4, 1, counter] = avg_performance
                    var_capacity_performance_hard[4, 1, counter] = avg_std


        #performance for different mu capacity soft threshold
        for counter, t in enumerate(soft_threshold):
            # best_s_thresh, performance = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, t, exp_name, type=1)
            best_s_thresh, avg_performance, avg_std = best_stepsize_wrt_thresh2(learn_mu, true_mu, converged_step_sizes, t, exp_name, type=1)

            # avg_performance = np.mean(performance[-100:])
            if data['params']['hidden_layers_mu'][0] == [8, 8]:
                if label=="Het":
                    mu_capacity_performance_soft[0, 0, counter] = avg_performance
                    var_capacity_performance_soft[0, 0, counter] = avg_std
                else:
                    mu_capacity_performance_soft[0, 1, counter] = avg_performance
                    var_capacity_performance_soft[0, 1, counter] = avg_std


            if data['params']['hidden_layers_mu'][0] == [12, 12]:
                if label=="Het":
                    mu_capacity_performance_soft[1, 0, counter] = avg_performance
                    var_capacity_performance_soft[1, 0, counter] = avg_std
                else:
                    mu_capacity_performance_soft[1, 1, counter] = avg_performance
                    var_capacity_performance_soft[1, 1, counter] = avg_std

            if data['params']['hidden_layers_mu'][0] == [16, 16]:
                if label=="Het":
                    mu_capacity_performance_soft[2, 0, counter] = avg_performance
                    var_capacity_performance_soft[2, 0, counter] = avg_std
                else:
                    mu_capacity_performance_soft[2, 1, counter] = avg_performance
                    var_capacity_performance_soft[2, 1, counter] = avg_std

            if data['params']['hidden_layers_mu'][0] == [24, 24]:
                if label=="Het":
                    mu_capacity_performance_soft[3, 0, counter] = avg_performance
                    var_capacity_performance_soft[3, 0, counter] = avg_std
                else:
                    mu_capacity_performance_soft[3, 1, counter] = avg_performance
                    var_capacity_performance_soft[3, 1, counter] = avg_std

            if file_name == ".DS_Store":
                continue
            if data['params']['hidden_layers_mu'][0] == [32, 32]:
                if label=="Het":
                    mu_capacity_performance_soft[4, 0, counter] = avg_performance
                    var_capacity_performance_soft[4, 0, counter] = avg_std
                else:
                    mu_capacity_performance_soft[4, 1, counter] = avg_performance
                    var_capacity_performance_soft[4, 1, counter] = avg_std

    plt.setp(axs_comp_hard, xticks=[0, 1, 2, 3, 4], xticklabels=['[8,8]', '[12,12]', '[16,16]', '[24,24]', '[32,32]'],)
    for counter, t in enumerate(hard_threshold):
        # axs_comp_hard[counter].plot(mu_capacity_performance_hard[:, 0, counter], label = "Het", color = "blue")
        drawPlotUncertainty([0,1,2,3,4], mu_capacity_performance_hard[:, 0, counter],
                            var_capacity_performance_hard[:, 0, counter],
                            label="Het", color="blue", axis=axs_comp_hard[counter])
        # axs_comp_hard[counter].plot(mu_capacity_performance_hard[:, 1, counter], label = "Reg", color = "red")
        drawPlotUncertainty([0,1,2,3,4], mu_capacity_performance_hard[:, 1, counter],
                            var_capacity_performance_hard[:, 1, counter],
                            label="Reg", color="red", axis=axs_comp_hard[counter])
        axs_comp_hard[counter].title.set_text("Hard_threshold:"+ str(t))
        axs_comp_hard[counter].spines['top'].set_visible(False)
        axs_comp_hard[counter].spines['right'].set_visible(False)
        axs_comp_hard[counter].legend()
        figs_comp_hard[counter].show()
        figs_comp_hard[counter].savefig("data/Plots/1/FailExample, hard-threshold:"+str(t)+".svg", format='svg')

    plt.setp(axs_comp_soft, xticks=[0, 1, 2, 3, 4], xticklabels=['[8,8]', '[12,12]', '[16,16]', '[24,24]', '[32,32]'],)
    for counter, t in enumerate(soft_threshold):
        # axs_comp_soft[counter].plot(mu_capacity_performance_soft[:, 0, counter], label = "Het", color = "blue")
        drawPlotUncertainty([0, 1, 2, 3, 4], mu_capacity_performance_soft[:, 0, counter],
                            var_capacity_performance_soft[:, 0, counter],
                            label="Het", color="blue", axis=axs_comp_soft[counter])
        # axs_comp_soft[counter].plot(mu_capacity_performance_soft[:, 1, counter], label = "Reg", color = "red")
        drawPlotUncertainty([0, 1, 2, 3, 4], mu_capacity_performance_soft[:, 1, counter],
                            var_capacity_performance_soft[:, 1, counter],
                            label="Reg", color="red", axis=axs_comp_soft[counter])
        axs_comp_soft[counter].title.set_text("Soft_threshold:"+ str(t))
        axs_comp_soft[counter].spines['top'].set_visible(False)
        axs_comp_soft[counter].spines['right'].set_visible(False)
        axs_comp_soft[counter].legend()
        figs_comp_soft[counter].show()
        figs_comp_soft[counter].savefig("data/Plots/1/FailExample, soft_threshold:"+str(t)+".svg", format='svg')

    for counter, fig in enumerate(figs_mu):
        fig.savefig("data/Plots/1/FailExampleMu"+str(counter)+".svg", format='svg')
    #     fig.show()

def hetfail2(hard_threshold, soft_threshold):
    with open('data/HetFail1/RegFail1_Irre_rangeUniformNoise3AAA.p', 'rb') as fp:
        reg_data = pickle.load(fp)

    path = "data/HetFail2"
    figs_mu, axs_mu = [], []
    figs_comp_hard, axs_comp_hard = [], []
    figs_comp_soft, axs_comp_soft = [], []
    for i in range(5):
        fig, ax = plt.subplots(1, 1, constrained_layout=False)
        figs_mu.append(fig)
        axs_mu.append(ax)
        fig, ax = plt.subplots(1, 1, constrained_layout=False)
        figs_comp_hard.append(fig)
        axs_comp_hard.append(ax)
        fig, ax = plt.subplots(1, 1, constrained_layout=False)
        figs_comp_soft.append(fig)
        axs_comp_soft.append(ax)

    gt = np.ones_like(axs_mu)
    results = os.listdir(path)

    mu_capacity_performance_hard = np.zeros([5, 2, len(hard_threshold)])
    mu_capacity_performance_soft = np.zeros([5, 2, len(soft_threshold)])
    for file_name in results:
        if 'Reg' in file_name:
            label = "Reg"
            color = "red"
        else:
            label = "Het"
            color = "blue"
        with open(path+"/"+file_name, 'rb') as fp:
            if file_name == ".DS_Store":
                continue
            data = pickle.load(fp)
        exp_name = file_name.split('.')[0]
        x, y, true_mu = recreate_dataset(exp_name, data['params'])
        learn_mu = data['avg_learn_mu']

        #learn mu plot
        best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data['mu_error_list'], exp_name)
        print(best_s_mse, file_name)
        index = int(file_name[-6])-1
        draw_learned_mu(data, axs_mu[index], color, label, best_s_mse, -1, x, y, het=label=="Het", ground_truth=gt[index])
        gt[index] = 0
        axs_mu[index].title.set_text(data['params']['hidden_layers_var'][0])

        #performance for different mu capacity hard threshold
        for counter, t in enumerate(hard_threshold):
            best_s_thresh, performance = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, t, exp_name, type=0)
            avg_performance = np.mean(performance[-100:])
            if data['params']['hidden_layers_var'][0] == [1]:
                if label=="Het":
                    mu_capacity_performance_hard[0, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_hard[0, 1, counter] = avg_performance

            if data['params']['hidden_layers_var'][0] == [2]:
                if label=="Het":
                    mu_capacity_performance_hard[1, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_hard[1, 1, counter] = avg_performance

            if data['params']['hidden_layers_var'][0] == [4]:
                if label=="Het":
                    mu_capacity_performance_hard[2, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_hard[2, 1, counter] = avg_performance

            if data['params']['hidden_layers_var'][0] == [8]:
                if label=="Het":
                    mu_capacity_performance_hard[3, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_hard[3, 1, counter] = avg_performance
            if file_name == ".DS_Store":
                continue
            if data['params']['hidden_layers_var'][0] == [16]:
                if label=="Het":
                    mu_capacity_performance_hard[4, 0, counter] = avg_performance
                else:
                    mu_capacity_performance_hard[4, 1, counter] = avg_performance

        #performance for different mu capacity soft threshold
        # for counter, t in enumerate(soft_threshold):
        #     best_s_thresh, performance = best_stepsize_wrt_thresh(learn_mu, true_mu, converged_step_sizes, t, exp_name, type=1)
        #     avg_performance = np.mean(performance[-100:])
        #     if data['params']['hidden_layers_var'][0] == [1]:
        #         if label=="Het":
        #             mu_capacity_performance_soft[0, 0, counter] = avg_performance
        #         else:
        #             mu_capacity_performance_soft[0, 1, counter] = avg_performance
        #
        #     if data['params']['hidden_layers_var'][0] == [2]:
        #         if label=="Het":
        #             mu_capacity_performance_soft[1, 0, counter] = avg_performance
        #         else:
        #             mu_capacity_performance_soft[1, 1, counter] = avg_performance
        #
        #     if data['params']['hidden_layers_var'][0] == [4]:
        #         if label=="Het":
        #             mu_capacity_performance_soft[2, 0, counter] = avg_performance
        #         else:
        #             mu_capacity_performance_soft[2, 1, counter] = avg_performance
        #
        #     if data['params']['hidden_layers_var'][0] == [8]:
        #         if label=="Het":
        #             mu_capacity_performance_soft[3, 0, counter] = avg_performance
        #         else:
        #             mu_capacity_performance_soft[3, 1, counter] = avg_performance
        #     if file_name == ".DS_Store":
        #         continue
        #     if data['params']['hidden_layers_var'][0] == [16]:
        #         if label=="Het":
        #             mu_capacity_performance_soft[4, 0, counter] = avg_performance
        #         else:
        #             mu_capacity_performance_soft[4, 1, counter] = avg_performance

    plt.setp(axs_comp_hard, xticks=[0, 1, 2, 3, 4], xticklabels=['1', '2', '4', '8', '16'],)
    for counter, t in enumerate(hard_threshold):
        axs_comp_hard[counter].plot(mu_capacity_performance_hard[:, 0, counter], label = "Het", color = "blue")
        axs_comp_hard[counter].plot(mu_capacity_performance_hard[:, 1, counter], label = "Reg", color = "red")
        axs_comp_hard[counter].title.set_text("Hard_threshold:"+ str(t))
        axs_comp_hard[counter].spines['top'].set_visible(False)
        axs_comp_hard[counter].spines['right'].set_visible(False)
        axs_comp_hard[counter].legend()
        figs_comp_hard[counter].show()
        # figs_comp_hard[counter].savefig("data/Plots/FailExample, threshold:"+str(t)+".eps", format='eps')

    # plt.setp(axs_comp_soft, xticks=[0, 1, 2, 3, 4], xticklabels=['1', '2', '4', '8', '16'],)
    # for counter, t in enumerate(soft_threshold):
    #     axs_comp_soft[counter].plot(mu_capacity_performance_soft[:, 0, counter], label = "Het", color = "blue")
    #     axs_comp_soft[counter].plot(mu_capacity_performance_soft[:, 1, counter], label = "Reg", color = "red")
    #     axs_comp_soft[counter].title.set_text("Soft_threshold:"+ str(t))
    #     axs_comp_soft[counter].spines['top'].set_visible(False)
    #     axs_comp_soft[counter].spines['right'].set_visible(False)
    #     axs_comp_soft[counter].legend()
    #     figs_comp_soft[counter].show()
    #     # figs_comp_soft[counter].savefig("data/Plots/FailExample, sigmoid par:"+str(t)+".eps", format='eps')

    for counter, fig in enumerate(figs_mu):
        # fig.savefig("data/Plots/FailExampleMu"+str(counter)+".eps", format='eps')
        fig.show()

def main5():
    path = "final_results"
    results = os.listdir(path)
    het_exps = [
                "MD_Bias_quadraticBias1_totall.p",
                "MD_Bias_rangelinearBias_totall.p",
                "MD_experiment_irreducible_error_single_y_totall.p",
                "MD_experiment_irreducible_linear_single_y_totall.p",
                "MD_experiment_quadraticBias_single_y_totall.p",
                "MD_experiment_rangeLBias_single_y_totall.P",
                "MD_Irre_linearNoise_totall.p",
                "MD_Irre_rangeUniformNoise_totall.p"
    ]

    reg_exps = [
                "MD_Bias_quadraticBias1_MSE_totall.p",
                "MD_Bias_rangelinearBias_MSE_totall.p",
                "MD_experiment_irreducible_error_single_y_MSE_totall.p",
                "MD_experiment_irreducible_linear_single_y_MSE_totall.p",
                "MD_experiment_quadraticBias_single_y_MSE_totall.p",
                "MD_experiment_rangeLBias_single_y_MSE_totall.p",
                "MD_Irre_linearNoise_MSE_totall.p",
                "MD_Irre_rangeUniformNoise_MSE_totall.p"
    ]

    for exp_name_het, exp_name_reg in zip(het_exps, reg_exps):
        print(exp_name_het, exp_name_reg)
        fig1, axs = plt.subplots(2, 2, constrained_layout=False)
        fig1.suptitle(exp_name_het.split("_")[1:-1])
        with open(path + "/" + exp_name_het, 'rb') as fp:
            try:
                data_het = pickle.load(fp)
            except:
                print("*****************CAN'T OPEN THE FILE:",exp_name_het)
                continue
        with open(path + "/" + exp_name_reg, 'rb') as fp:
            try:
                data_reg = pickle.load(fp)
            except:
                print("*****************CAN'T OPEN THE FILE:",exp_name_reg)
                continue

        exit(0)
        # mu mse plot
        i, j = 0, 0
        best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data_reg['mu_error_list'], exp_name_reg)
        step_size = np.log2(data_reg['params']['step_sizes'][best_s_mse])
        draw_mse_stepsize(data_reg, axs[i, j], 'red', 'reg' + str(step_size), best_s_mse, title=False)

        best_s_mse, converged_step_sizes = best_stepsize_wrt_mse(data_het['mu_error_list'], exp_name_het)
        step_size = np.log2(data_het['params']['step_sizes'][best_s_mse])
        draw_mse_stepsize(data_het, axs[i, j], 'blue', 'het' + str(step_size), best_s_mse, title=False)
        axs[i, j].set_title("mse mu")

        i, j = 0, 1
        axs[i, j].set_yscale("log")
        draw_var_mse_stepsize(data_het, axs[i, j], 'blue', 'het' + str(step_size), best_s_mse, title=False)
        axs[i, j].set_title("mse var")

        i, j = 1, 0
        axs[i, j].set_title("hard thresh")

        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].spines['right'].set_visible(False)
        learn_mu_reg = data_reg['last_mu']
        learn_mu_het = data_het['last_mu']
        exp_name = exp_name_het
        x, y, true_mu = recreate_dataset(exp_name, data_het['params'])

        # best parameters plot hard threshold
        best_s_mse_het, converged_step_sizes_het = best_stepsize_wrt_mse(data_het['mu_error_list'], "het")
        best_s_mse_reg, converged_step_sizes_reg = best_stepsize_wrt_mse(data_reg['mu_error_list'], "reg")

        threshold = np.arange(0.0, 0.001, 0.00001)
        threshold = np.concatenate((threshold, np.arange(0.001, 0.1, 0.01)))
        threshold = np.concatenate((threshold, np.arange(0.1, 1.0, 0.1)))

        avg_performance_list = []
        std_performance_list = []
        for t in threshold:
            best_s_thresh, avg_performance, std_performance = best_stepsize_wrt_thresh2(learn_mu_het, true_mu,
                                                                                        converged_step_sizes_het, t, exp_name_het,
                                                                                        type=0)

            avg_performance_list.append(avg_performance)
            std_performance_list.append(std_performance)
        axs[i, j].set_xscale("log")

        drawPlotUncertainty(threshold, np.asarray(avg_performance_list), np.asarray(std_performance_list)
                            , label="het", color="blue", axis=axs[i, j])

        avg_performance_list = []
        std_performance_list = []
        for t in threshold:
            best_s_thresh, avg_performance, std_performance = best_stepsize_wrt_thresh2(learn_mu_reg, true_mu,
                                                                                        converged_step_sizes_reg, t, exp_name_reg,
                                                                                        type=0)

            avg_performance_list.append(avg_performance)
            std_performance_list.append(std_performance)

        drawPlotUncertainty(threshold, np.asarray(avg_performance_list), np.asarray(std_performance_list)
                            , label="reg", color="red", axis=axs[i, j])


        i, j = 1, 1
        axs[i, j].set_title("soft thresh")
        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].spines['right'].set_visible(False)
        # best parameters plot soft threshold
        threshold = np.arange(0, 128, 1)
        avg_performance_list = []
        std_performance_list = []
        for t in threshold:
            best_s_thresh, avg_performance, std_performance = best_stepsize_wrt_thresh2(learn_mu_het, true_mu,
                                                                                        converged_step_sizes_het, t, exp_name_het,
                                                                                        type=1)
            avg_performance_list.append(avg_performance)
            std_performance_list.append(std_performance)
        # ax_soft.plot(threshold, avg_performance_list, label=label, color=color)
        # ax_soft.set_xscale("log")
        drawPlotUncertainty(threshold, np.asarray(avg_performance_list), np.asarray(std_performance_list)
                            , label="het", color="blue", axis=axs[i, j])
        avg_performance_list = []
        std_performance_list = []
        for t in threshold:
            best_s_thresh, avg_performance, std_performance = best_stepsize_wrt_thresh2(learn_mu_reg, true_mu,
                                                                                        converged_step_sizes_reg, t,
                                                                                        exp_name_reg,
                                                                                        type=1)
            avg_performance_list.append(avg_performance)
            std_performance_list.append(std_performance)
        # ax_soft.plot(threshold, avg_performance_list, label=label, color=color)
        # ax_soft.set_xscale("log")
        drawPlotUncertainty(threshold, np.asarray(avg_performance_list), np.asarray(std_performance_list)
                            , label="reg", color="red", axis=axs[i, j])
        fig1.show()

if __name__ == "__main__":
    threshold_hard = [0.1, 0.5, 1.0]
    threshold_soft = [32, 16, 8]
    path = [
           # "data/Irre_rangeUniformNoise", \
           # "data/Bias_rangelinearBias", \
           # "data/Bias_quadraticBias", \
           # "data/Irre_LinearNoise", \
           # "data/fixedMu", \
           # "data/expwitherror/fixed_mu",
           "data/expwitherror/Bias_quadraticBias1",
           "data/expwitherror/Bias_rangelinearBias",
           "data/expwitherror/Irre_linearNoise",
           "data/expwitherror/Irre_rangeUniformNoise"
    ]
    main5()
    # file_name= "data/IrreBias_fixedMu_linearNoise_fixedBiasAAA.p"
    # with open(file_name, 'rb') as fp:
    #     data = pickle.load(fp)
    # print(data['sigma_error_list'].shape)
    # fig, axs = plt.subplots(1, 1, constrained_layout=False)
    #
    # draw_var_mse_stepsize(data, axs, color="blue", label="test", step_size_index=0, title=False)
    # fig.show()
    # for p in path:
        # main(p)
        # main4(p)
        # fixedMu(p)
        # main2(p)
        # main4(p)


    # main3(threshold_hard, threshold_soft)
    # hetfail1(threshold_hard, threshold_soft)
    # hetfail2(threshold_hard, threshold_soft)
