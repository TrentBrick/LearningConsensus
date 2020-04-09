import argparse
import numpy as np
from sklearn.model_selection import ParameterGrid
import multiprocessing
#from test import *
import torch
#import main
import pandas as pd
#import gym 
import consensus_env
import ppo_code.ppo as ppo
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


def initialize_parameters():
    parser = argparse.ArgumentParser()
    # Experiment Settings
    parser.add_argument("--exp_name", type=str, action='store', nargs='+', default=["TestRun"], help="name of experiment")
    parser.add_argument("--directory", type=str, action='store', nargs='+', default = ["runs/"], help='directory to save results in')
    parser.add_argument("--random_seed", type=int, action='store', nargs='+', default = [27], help='seed to start the simulation from')
    parser.add_argument("--ncores", type=int, action='store', nargs='+', default = [1], help='number of cores to use. if -1 then it uses all of them. ')

    parser.add_argument("--load_policy_honest", type=str, action='store', nargs='+', default=['None'], help='path to load in a pretrained policy for honest')
    parser.add_argument("--load_policy_byz", type=str, action='store', nargs='+', default=['None'], help='path to load in a pretrained policy for honest')
    parser.add_argument("--LOAD_PATH_EXPERIMENT", type=str, action='store', nargs='+', default = ['saved_models/'], help='Path to the saved policies')
    #parser.add_argument("--honest_policy_LOAD_PATH", type=str, action='store', nargs='+', default = [''], help='Path to the saved honest')
    #parser.add_argument("--byz_policy_LOAD_PATH", type=str, action='store', nargs='+', default = [''], help='Path to the saved byzantine')
    parser.add_argument("--train_honest", type=buildBool, action='store', nargs='+', default = [True], help='Can ensure that the honest are not trained. ')
    parser.add_argument("--train_byz", type=buildBool, action='store', nargs='+', default = [True], help='Can ensure that the byz are not trained. ')
    parser.add_argument("--byz_honest_train_ratio", type=int, action='store', nargs='+', default = [1], help='Ratio of epochs we train byzantine for 1 honest. A value of 1 means has no ratio training')

    # Environment Settings
    parser.add_argument("--scenario", type=str, action='store', nargs='+', default = ['Basic'], help='')
    parser.add_argument("--commit_vals", action ='store', type=str, default = ['(0,1)'], nargs='+', help="Commit values. -commit_vals (0,1) (2,0)")
    parser.add_argument("--num_agents", type=int, action='store', nargs='+', default = [3], help='overall number of agents in simulation')
    parser.add_argument("--num_byzantine", type=int, action='store', nargs='+', default = [0], help='overall number of byzantine agents in simulation')
    #parser.add_arguemnt("--action_space", type=int, action='store', nargs='+', default=[0,2], help='actions that agent can take - default is send init value and commit to a value')
    
    # Training Settings
    parser.add_argument("--epochs", type=int, action='store', nargs='+', default = [700], help='number of epochs')
    parser.add_argument("--actions_per_epoch", type=int, action='store', nargs='+', default = [1000], help='number of protocol simulations per epoch')
    parser.add_argument("--max_round_len", type=int, action='store', nargs='+', default = [50], help='limit on the number of rounds per protocol simulation')
    parser.add_argument("--print_every", type=int, action='store', nargs='+', default = [5], help='')

    # RL Settings
    parser.add_argument("--use_PKI", type=buildBool, action='store', nargs='+', default = [False], help='Use Public Key Infrastructure?')
    parser.add_argument("--use_vpg", type=buildBool, action='store', nargs='+', default = [False], help='if False will use REINFORCE')
    parser.add_argument("--vpg_epochs_ratio", type=int, action='store', nargs='+', default = [2], help='Ratio of epochs we train only the vpg functions and not reinforce. A value of 1 means 1:1')
    
    parser.add_argument("--honest_starting_temp", type=float, action='store', nargs='+', default = [6.0], help='starting temperature')
    parser.add_argument("--byz_starting_temp", type=float, action='store', nargs='+', default = [6.0], help='starting temperature')
    parser.add_argument("--temp_anneal", type=float, action='store', nargs='+', default = [0.985], help='rate at which the temperature anneals per epoch')
    parser.add_argument("--temp_fix_point", type=float, action='store', nargs='+', default = [1.0], help='point at which temperature will stop annealing or if heat jumps are on, the point at which the temperature will bounce back to its starting point')
    parser.add_argument("--honest_can_send_either_value", type=buildBool, action='store', nargs='+', default = [False], help='can the honest agents send only their init value or other values also?')
    parser.add_argument("--use_heat_jumps", type=buildBool, action='store', nargs='+', default = [False], help='when it hits the temp fix point, increase the temp back to the starting temp')
    
    parser.add_argument("--rl_algo_wanted", type=str, action='store', nargs='+', default = ['vpg'], help='')
    parser.add_argument("--gamma", type=float, action='store', nargs='+', default = [0.99], help='')
    parser.add_argument("--lam", type=float, action='store', nargs='+', default = [0.95], help='')
    parser.add_argument("--clip_ratio", type=float, action='store', nargs='+', default = [0.2], help='')
    parser.add_argument("--vf_lr", type=float, action='store', nargs='+', default = [0.001], help='')
    parser.add_argument("--train_policy_iters", type=int, action='store', nargs='+', default = [80], help='')
    parser.add_argument("--train_vf_iters", type=int, action='store', nargs='+', default = [80], help='')
    parser.add_argument("--target_kl", type=float, action='store', nargs='+', default = [0.01], help='')
    # parser.add_argument("--logger_kwargs", type=dict(), action='store', nargs='+', default = [dict()], help='')
    parser.add_argument("--save_freq", type=int, action='store', nargs='+', default = [10], help='')

    ## Penalties for rewards
    parser.add_argument("--send_all_first_round_reward", action ='store', type=float, default = [0.3], nargs='+')
    parser.add_argument("--consistency_violation", action ='store', type=float, default = [-1.0], nargs='+', help='from the perspective of the honest. The inverse is applied to the Byzantine')
    parser.add_argument("--validity_violation", action ='store', type=float, default = [-2.0], nargs='+')
    parser.add_argument("--majority_violation", action ='store', type=float, default = [-1.0], nargs='+')
    parser.add_argument("--correct_commit", action ='store', type=float, default = [1], nargs='+')
    parser.add_argument("--additional_round_penalty", action ='store', type=float, default = [-0.03], nargs='+')
    parser.add_argument("--termination_penalty", action ='store', type=float, default = [-.0], nargs='+')


    #parser.add_argument("--consistency_violation", action ='store', type=str, default = [-1,1], nargs='+')
    #parser.add_argument("--validity_violation", action ='store', type=str, default = [-1,1], nargs='+')
    #parser.add_argument("--majority_violation", action ='store', type=str, default = [-0.5,0.5], nargs='+')
    #parser.add_argument("--correct_commit", action ='store', type=str, default = [1,-1], nargs='+')
    
    ## NN Settings
    parser.add_argument("--learning_rate", type=float, action='store', nargs='+', default = [0.003], help='')
    parser.add_argument("--batch_size", type=int, action='store', nargs='+', default = [32], help='')
    parser.add_argument("--hidden_sizes", action ='store', type=str, default = ['(16,8)'], nargs='+', help = "Hidden sizes of neural net. -hidden_sizes (16,8) (2,3)")
    parser.add_argument("--activation", type=getActivation, action='store', nargs='+', default = [torch.tanh], help='Activation functions: tanh, relu, sigmoid')
    parser.add_argument("--output_activation", type=getActivation, action='store', nargs='+', default = [None], help='')
    parser.add_argument("--use_bias", type=buildBool, action='store', nargs='+', default = [True], help='')
    parser.add_argument("--starting_ep", type=int, action='store', nargs='+', default = [1], help='')
    parser.add_argument("--null_message_val", type=int, action='store', nargs='+', default = [2], help='')
    ###This value will be set as a function of commit_vals and num_agents

    args = parser.parse_args()

    args.hidden_sizes = buildTuple(args.hidden_sizes)
    args.commit_vals = buildTuple(args.commit_vals)
    '''args.consistency_violation = buildNPArray(args.consistency_violation)
    args.validity_violation = buildNPArray(args.validity_violation)
    args.majority_violation = buildNPArray(args.majority_violation)
    args.correct_commit = buildNPArray(args.correct_commit)'''
    # print(args)

    if args.ncores == -1:
        args.ncores = multiprocessing.cpu_count()

    #args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Create permutation matrix
    arg_dict = vars(args)
    tot_combos = 1
    ##Create global variables
    for v in arg_dict.values():
        tot_combos *= len(v)
    pg = ParameterGrid(arg_dict)

    res = pd.DataFrame.from_dict(pg)
    res['timestamp'] = np.zeros(res.shape[0])
    res['last_honest_win'] = np.zeros(res.shape[0])
    res['honest_50'] =np.zeros(res.shape[0])
    res['honest_75']=np.zeros(res.shape[0])
    res['honest_90']=np.zeros(res.shape[0])

    for i in range(tot_combos):
        print(' ====================== Running param combo ', i+1, '/', tot_combos, '======================')
        print('combo of params is:', pg[i])

        env = consensus_env.ConsensusEnv(pg[i])
        # should I be using gym.make here??

        ##Fork 
        mpi_fork(pg[i]['ncores'])
        # TODO: CHANGE core.MLPActorCritic when NN hidden layers are changed
        ppo.ppo_algo(env, gamma=pg[i]['gamma'], 
            seed=pg[i]['random_seed'], actions_per_epoch=pg[i]['actions_per_epoch'], 
            epochs=pg[i]['epochs'])
            #logger_kwargs=logger_kwargs)
        
        '''
        #receiving back results to store so that multiple iterations can be compared:
        exp_dir, timestamp, last_honest_win, honest_90, honest_75, honest_50 = main.main(pg[i])  
        res.loc[i, 'timestamp'] = timestamp
        res.loc[i, 'last_honest_win'] = last_honest_win
        res.loc[i, 'honest_50'] = honest_50
        res.loc[i, 'honest_75'] = honest_75
        res.loc[i, 'honest_90'] = honest_90
    
        res.to_csv(exp_dir + "ParamCombos.csv")'''

def buildBool(arg):
    # convert strings to booleans
    # this function is needed as there is a bug with argparse 
    # where if your arg is by default 'False' and you then set it in
    # the command line to 'False' then it will evaluate to being True
    
    print('running build bool', arg)
    if arg == 'False':
        return False
    else:
        return True

def buildTuple(argument):
    count = 0
    values = []
    curr_tuple = ()
    print(argument)
    for val in argument:
        val = val.replace("(", ",")
        val = val.replace(")", ",")
        val = val.split(",")
        print(val)
        for element in range(0, len(val)):
            if val[element] is not "":
                curr_tuple = curr_tuple + (int(val[element]),)
        values.append(curr_tuple)
        curr_tuple = ()
        count = count + 1
    print (values)
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

def getActivation(activation_string):
    print(' is this running at all?!?!?!')
    print('activaiton string is: ', activation_string)
    if activation_string is 'tanh':
        print('reuturning tanh')
        return torch.tanh
    if activation_string is 'relu':
        return torch.relu
    if activation_string is 'sigmoid':
        return torch.sigmoid
    if activation_string is 'hardtanh':
        return torch.hardtanh
    if activation_string is 'leakyrelu':
        return torch.leakyrelu
    else:
        return None

if __name__=='__main__':
    initialize_parameters()
