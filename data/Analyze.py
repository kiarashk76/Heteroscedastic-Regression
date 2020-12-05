import pickle
import numpy as np
import os
from Exp_Bias import *
from Exp_IrreducibleError import *
import matplotlib.pyplot as plt

results = os.listdir("data")
exp_map_name = {
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
}
for file_name in results:
    if file_name == "Analyze.py":
        continue
    with open('data/'+file_name, 'rb') as fp:
        data = pickle.load(fp)
    exp_name = file_name.split('.')[0]
    exp = exp_map_name[exp_name](data['params'], exp_name)
    np.random.seed(0)
    exp.create_dataset()

    print(data['avg_learn_mu'].shape)


