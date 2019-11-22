import argparse
import numpy as np
from sklearn.model_selection import ParameterGrid
from test import *
import torch
import main

def initialize_parameters():
    parser = argparse.ArgumentParser()
    # Experiment Settings
    parser.add_argument("--experiment_base_name", type=str, action='store', nargs='+', default=["TestRun"], help="name of experiment")
    parser.add_argument("--directory", type=str, action='store', nargs='+', default = ["runs/"], help='directory to save results in')
    parser.add_argument("--random_seed", type=int, action='store', nargs='+', default = [27], help='seed to start the simulation from')

    parser.add_argument("--load_policy_honest", type=bool, action='store', nargs='+', default=[False], help='load in a pretrained policy for honest')
    parser.add_argument("--load_policy_byz", type=bool, action='store', nargs='+', default=[False], help='load in a pretrained policy for honest')
    parser.add_argument("--LOAD_PATH_EXPERIMENT", type=str, action='store', nargs='+', default = ['saved_models/'], help='Path to the saved policies')
    parser.add_argument("--honest_policy_LOAD_PATH", type=str, action='store', nargs='+', default = [''], help='Path to the saved honest')
    parser.add_argument("--byz_policy_LOAD_PATH", type=str, action='store', nargs='+', default = [''], help='Path to the saved byzantine')
    parser.add_argument("--train_honest", type=bool, action='store', nargs='+', default = [True], help='Can ensure that the honest are not trained. ')
    parser.add_argument("--train_byz", type=bool, action='store', nargs='+', default = [True], help='Can ensure that the byz are not trained. ')
    parser.add_argument("--byz_honest_train_ratio", type=int, action='store', nargs='+', default = [1], help='Ratio of epochs we train byzantine for 1 honest. a value of 1 means 1:1')

    # Environment Settings
    parser.add_argument("--scenario", type=str, action='store', nargs='+', default = ['Basic'], help='')
    parser.add_argument("-commit_vals", action ='store', type=str, default = (0,1), nargs='+', help="Commit values. -commit_vals (0,1) (2,0)")
    parser.add_argument("--num_agents", type=int, action='store', nargs='+', default = [3], help='overall number of agents in simulation')
    parser.add_argument("--num_byzantine", type=int, action='store', nargs='+', default = [0], help='overall number of byzantine agents in simulation')

    # Training Settings
    parser.add_argument("--epochs", type=int, action='store', nargs='+', default = [400], help='number of epochs')
    parser.add_argument("--iters_per_epoch", type=int, action='store', nargs='+', default = [200], help='number of protocol simulations per epoch')
    parser.add_argument("--max_round_len", type=int, action='store', nargs='+', default = [10], help='limit on the number of rounds per protocol simulation')
    parser.add_argument("--print_every", type=int, action='store', nargs='+', default = [5], help='')

    # RL Settings
    parser.add_argument("--use_PKI", type=bool, action='store', nargs='+', default = [False], help='Use Public Key Infrastructure?')
    parser.add_argument("--use_vpg", type=bool, action='store', nargs='+', default = [False], help='if False will use REINFORCE')
    parser.add_argument("--honest_starting_temp", type=float, action='store', nargs='+', default = [6.0], help='starting temperature')
    parser.add_argument("--byz_starting_temp", type=float, action='store', nargs='+', default = [6.0], help='starting temperature')
    parser.add_argument("--temp_anneal", type=float, action='store', nargs='+', default = [0.985], help='')
    parser.add_argument("--temp_fix_point", type=float, action='store', nargs='+', default = [1.0], help='')
    parser.add_argument("--honest_can_send_either_value", type=bool, action='store', nargs='+', default = [False], help='can the honest agents send only their init value or other values also?')
    parser.add_argument("--use_heat_jumps", type=bool, action='store', nargs='+', default = [False], help='when it hits the temp fix point, increase the temp back to the starting temp')
    parser.add_argument("--rl_algo_wanted", type=str, action='store', nargs='+', default = ['vpg'], help='')
    parser.add_argument("--steps_per_epoch", type=int, action='store', nargs='+', default = [4000], help='')
    parser.add_argument("--gamma", type=float, action='store', nargs='+', default = [0.999], help='')
    parser.add_argument("--clip_ratio", type=float, action='store', nargs='+', default = [0.2], help='')
    parser.add_argument("--vf_lr", type=float, action='store', nargs='+', default = [0.001], help='')
    parser.add_argument("--train_policy_iters", type=int, action='store', nargs='+', default = [80], help='')
    parser.add_argument("--train_vf_iters", type=int, action='store', nargs='+', default = [80], help='')
    parser.add_argument("--lam", type=float, action='store', nargs='+', default = [0.97], help='')
    parser.add_argument("--target_kl", type=float, action='store', nargs='+', default = [0.01], help='')
    # parser.add_argument("--logger_kwargs", type=dict(), action='store', nargs='+', default = [dict()], help='')
    parser.add_argument("--save_freq", type=int, action='store', nargs='+', default = [10], help='')

    ## Penalties for rewards
    parser.add_argument("--send_all_first_round_reward", action ='store', type=str, default = [0.3], nargs='+')
    parser.add_argument("--consistency_violation", action ='store', type=str, default = [-1,1], nargs='+')
    parser.add_argument("--validity_violation", action ='store', type=str, default = [-4,1], nargs='+')
    parser.add_argument("--majority_violation", action ='store', type=str, default = [-0.5,0.5], nargs='+')
    parser.add_argument("--correct_commit", action ='store', type=str, default = [1,-1], nargs='+')
    parser.add_argument("--additional_round_penalty", action ='store', type=str, default = [-0.03], nargs='+')

    ## NN Settings
    parser.add_argument("--learning_rate", type=float, action='store', nargs='+', default = [0.003], help='')
    parser.add_argument("--batch_size", type=int, action='store', nargs='+', default = [32], help='')
    parser.add_argument("--hidden_sizes", action ='store', type=str, default = (16,8), nargs='+', help = "Hidden sizes of neural net. -hidden_sizes (16,8) (2,3)")
    parser.add_argument("--activation", type=str, action='store', nargs='+', default = ["tanh"], help='Activation functions - tanh, relu, sigmoid')
    parser.add_argument("--output_activation", type=None, action='store', nargs='+', default = [None], help='')
    parser.add_argument("--use_bias", type=bool, action='store', nargs='+', default = [True], help='')
    parser.add_argument("--starting_ep", type=int, action='store', nargs='+', default = [1], help='')
    parser.add_argument("--null_message_val", type=int, action='store', nargs='+', default = [2], help='')
    ###This value will be set as a function of commit_vals and num_agents

    args = parser.parse_args()

    args.hidden_sizes = buildTuple(args.hidden_sizes)
    args.commit_vals = buildTuple(args.commit_vals)
    args.consistency_violation = buildNPArray(args.consistency_violation)
    args.validity_violation = buildNPArray(args.validity_violation)
    args.majority_violation = buildNPArray(args.majority_violation)
    args.correct_commit = buildNPArray(args.correct_commit)
    # print(args)

    ## Create permutation matrix
    arg_dict = vars(args)
    tot_combos = 1
    ##Create global variables
    for v in arg_dict.values():
        tot_combos *= len(v)
    pg = ParameterGrid(arg_dict)
    for i in range(tot_combos):
        print(' ====================== Running param combo ', i, '/', tot_combos, '======================')
        print('combo of params is:', pg[i])
        main.main(pg[i])  

def buildTuple(argument):
    count = 0
    values = []
    curr_tuple = ()
    for val in argument:
        curr_tuple = curr_tuple + (int(val),)
        count+=1
        if count%2 == 0:
            values.append(curr_tuple)
            curr_tuple = ()
    return values

def buildNPArray(argument):
    count = 0
    values = []
    curr_array = []
    for val in argument:
        curr_array.append(int(val))
        count+=1
        if count%2 == 0:
            values.append(np.array(curr_array))
            curr_array = []
    return values

if __name__=='__main__':
    initialize_parameters()
