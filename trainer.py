from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys
#import config
#from model import make_model, simulate
from es import CMAES, SimpleGA, OpenES, PEPG
import argparse
import time
import cma
from controller_model import Models, load_parameters, flatten_parameters
### ES related code
num_episode = 1
eval_steps = 25 # evaluate every N_eval steps
retrain_mode = True
cap_time_mode = True

num_worker = 8
num_worker_trial = 16

population = num_worker * num_worker_trial

scenario = 'carracing'
optimizer = 'cma'
antithetic = True
batch_mode = 'mean'

# seed for reproducibility
seed_start = 0

### name of the file (can override):
filebase = None

game = None
model = None
num_params = -1

es = None

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

PRECISION = 10000
SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
RESULT_PACKET_SIZE = 4*num_worker_trial
###

def initialize_settings(parameters, sigma_init=0.1, sigma_decay=0.9999):
    global population, filebase, game, model, num_params, es, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE
    population = num_worker * num_worker_trial
    filebase = 'es_log/'+scenario+'.'+optimizer+'.'+str(num_episode)+'.'+str(population)
    #game = config.games[scenario]

    # TODO: init the controller model. and VAE and MDRNN. 
    model = Models(mdir = 'exp_dir', parameters=parameters) # time limit
    num_params = len( flatten_parameters(model.policy.parameters()) )
    print("size of model", num_params)

    if optimizer == 'ses':
        ses = PEPG(num_params,
            sigma_init=sigma_init,
            sigma_decay=sigma_decay,
            sigma_alpha=0.2,
            sigma_limit=0.02,
            elite_ratio=0.1,
            weight_decay=0.005,
            popsize=population)
        es = ses
    elif optimizer == 'ga':
        ga = SimpleGA(num_params,
            sigma_init=sigma_init,
            sigma_decay=sigma_decay,
            sigma_limit=0.02,
            elite_ratio=0.1,
            weight_decay=0.005,
            popsize=population)
        es = ga
    elif optimizer == 'cma':
        cma = CMAES(num_params,
            sigma_init=sigma_init,
            popsize=population)
        es = cma
        #es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), sigma_init,
        #                        {'popsize': population})
    elif optimizer == 'pepg':
        pepg = PEPG(num_params,
            sigma_init=sigma_init,
            sigma_decay=sigma_decay,
            sigma_alpha=0.20,
            sigma_limit=0.02,
            learning_rate=0.01,
            learning_rate_decay=1.0,
            learning_rate_limit=0.01,
            weight_decay=0.005,
            popsize=population)
        es = pepg
    else:
        oes = OpenES(num_params,
            sigma_init=sigma_init,
            sigma_decay=sigma_decay,
            sigma_limit=0.02,
            learning_rate=0.01,
            learning_rate_decay=1.0,
            learning_rate_limit=0.01,
            antithetic=antithetic,
            weight_decay=0.005,
            popsize=population)
        es = oes

    PRECISION = 10000
    SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
    RESULT_PACKET_SIZE = 4*num_worker_trial
###

def sprint(*args):
    print(*args) # if python3, can do print(*args)
    sys.stdout.flush()

class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2**31-1)
    def next_seed(self):
        result = np.random.randint(self.limit)
        return result
    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result

def encode_solution_packets(seeds, solutions, train_mode=1, max_len=-1):
    n = len(seeds)
    result = []
    worker_num = 0
    for i in range(n):
        worker_num = int(i / num_worker_trial) + 1
        result.append([worker_num, i, seeds[i], train_mode, max_len])
        result.append(np.round(np.array(solutions[i])*PRECISION,0))
    result = np.concatenate(result).astype(np.int32)
    #print('result in encode solution packets!!! pre split', len(result))
    result = np.split(result, num_worker)
    #print('result in encode solution packets!!!', len(result))
    return result

def decode_solution_packet(packet):
    packets = np.split(packet, num_worker_trial)
    result = []
    for p in packets:
        result.append([p[0], p[1], p[2], p[3], p[4], p[5:].astype(np.float)/PRECISION])
    return result

def encode_result_packet(results):
    r = np.array(results)
    r[:, 2:4] *= PRECISION
    return r.flatten().astype(np.int32)

def decode_result_packet(packet):
    r = packet.reshape(num_worker_trial, 4)
    workers = r[:, 0].tolist()
    jobs = r[:, 1].tolist()
    fits = r[:, 2].astype(np.float)/PRECISION
    fits = fits.tolist()
    times = r[:, 3].astype(np.float)/PRECISION
    times = times.tolist()
    result = []
    n = len(jobs)
    for i in range(n):
        result.append([workers[i], jobs[i], fits[i], times[i]])
    return result

def worker(weights, seed, train_mode_int=1, max_len=-1):

    train_mode = (train_mode_int == 1)
    #model.set_model_params(weights) feeding into simulate. 
    # TODO: run the simulation here. need to return the rewards and the end times of each. 
    #sprint('starting worker simulation, seed', seed)
    reward_list, t_list = model.simulate(weights, train_mode=train_mode, render_mode=False, 
        num_episode=num_episode, seed=seed)
    #sprint('finished worker simulations, seed', seed)
    #if np.max(reward_list) >0:
    #    print('correct performance!!!')
    #print('rewards',reward_list,' of worker seed:', seed)
    if batch_mode == 'min':
        reward = np.min(reward_list)
    else:
        reward = np.mean(reward_list)
    t = np.mean(t_list)
    return reward, t

def slave():
    # TODO: make the gym environment for each of the slaves. Do I need to though? already inside model.init()..?
    model.make_env()
    packet = np.empty(SOLUTION_PACKET_SIZE, dtype=np.int32)
    while 1:
        comm.Recv(packet, source=0)
        assert(len(packet) == SOLUTION_PACKET_SIZE)
        solutions = decode_solution_packet(packet)
        results = []
        for solution in solutions:
            worker_id, jobidx, seed, train_mode, max_len, weights = solution
            assert (train_mode == 1 or train_mode == 0), str(train_mode)
            worker_id = int(worker_id)
            possible_error = "work_id = " + str(worker_id) + " rank = " + str(rank)
            assert worker_id == rank, possible_error
            jobidx = int(jobidx)
            seed = int(seed)
            fitness, timesteps = worker(weights, seed, train_mode, max_len)
            results.append([worker_id, jobidx, fitness, timesteps])
        result_packet = encode_result_packet(results)
        assert len(result_packet) == RESULT_PACKET_SIZE
        comm.Send(result_packet, dest=0)

def send_packets_to_slaves(packet_list):
    num_worker = comm.Get_size()
    #print('=========================', len(packet_list), num_worker,  num_worker-1)
    assert len(packet_list) == num_worker-1
    for i in range(1, num_worker):
        packet = packet_list[i-1]
        assert(len(packet) == SOLUTION_PACKET_SIZE)
        comm.Send(packet, dest=i)

def receive_packets_from_slaves():
    result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)

    reward_list_total = np.zeros((population, 2))

    check_results = np.ones(population, dtype=np.int)
    for i in range(1, num_worker+1):
        comm.Recv(result_packet, source=i)
        results = decode_result_packet(result_packet)
        for result in results:
            worker_id = int(result[0])
            possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
            assert worker_id == i, possible_error
            idx = int(result[1])
            reward_list_total[idx, 0] = result[2]
            reward_list_total[idx, 1] = result[3]
            check_results[idx] = 0

    check_sum = check_results.sum()
    assert check_sum == 0, check_sum
    return reward_list_total

def evaluate_batch(model_params, max_len=-1):
    # duplicate model_params
    solutions = []
    for i in range(es.popsize):
        solutions.append(np.copy(model_params))

    seeds = np.arange(es.popsize)

    packet_list = encode_solution_packets(seeds, solutions, train_mode=0, max_len=max_len)

    send_packets_to_slaves(packet_list)
    reward_list_total = receive_packets_from_slaves()

    reward_list = reward_list_total[:, 0] # get rewards
    return np.mean(reward_list)

def master():

    start_time = int(time.time())
    sprint("training", scenario)
    sprint("population", es.popsize)
    sprint("num_worker", num_worker)
    sprint("num_worker_trial", num_worker_trial)
    sys.stdout.flush()

    seeder = Seeder(seed_start)

    filename = filebase+'.json'
    filename_log = filebase+'.log.json'
    filename_hist = filebase+'.hist.json'
    filename_best = filebase+'.best.json'

    # TODO: make enviornment here. Actually do I need to? This master isnt doing anything... 
    model.make_env()

    t = 0

    history = []
    eval_log = []
    best_reward_eval = 0
    best_model_params_eval = None

    max_len = -1 # max time steps (-1 means ignore)

    while True:
        t += 1

        #sprint('asking for solutions in master')
        solutions = es.ask()
        #sprint('============================= solutions from es.ask', solutions.shape)
        #sprint('getting seeds')
        if antithetic:
            seeds = seeder.next_batch(int(es.popsize/2))
            seeds = seeds+seeds
        else:
            seeds = seeder.next_batch(es.popsize)

        #sprint('encoding solution packets')
        packet_list = encode_solution_packets(seeds, solutions, max_len=max_len)

        send_packets_to_slaves(packet_list)
        reward_list_total = receive_packets_from_slaves()

        reward_list = reward_list_total[:, 0] # get rewards

        mean_time_step = int(np.mean(reward_list_total[:, 1])*100)/100. # get average time step
        max_time_step = int(np.max(reward_list_total[:, 1])*100)/100. # get average time step
        avg_reward = int(np.mean(reward_list)*100)/100. # get average time step
        std_reward = int(np.std(reward_list)*100)/100. # get average time step

        #sprint('reward list being put into es.tell', len(reward_list), reward_list)
        es.tell(reward_list)

        es_solution = es.result()
        model_params = es_solution[0] # best historical solution
        reward = es_solution[1] # best reward
        curr_reward = es_solution[2] # best of the current batch

        # TODO: update the model parameters here. Why are they quantized and set here? 
        #sprint('master model controller about to load in')
        model.policy = load_parameters(np.array(model_params).round(4), model.policy)
        #sprint('loaded in master model controller')
        r_max = int(np.max(reward_list)*100)/100.
        r_min = int(np.min(reward_list)*100)/100.

        curr_time = int(time.time()) - start_time

        h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., max_time_step+1.)

        if cap_time_mode:
            max_len = 2*int(mean_time_step+1.0)
        else:
            max_len = -1

        history.append(h)

        with open(filename, 'wt') as out:
            res = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

        with open(filename_hist, 'wt') as out:
            res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

        sprint('iteration, wall time, avg rew, min, max, std, time things, mean and max time steps')
        sprint('================================',scenario, h)

        if (t == 1):
            best_reward_eval = avg_reward
        if (t % eval_steps == 0): # evaluate on actual task at hand

            prev_best_reward_eval = best_reward_eval
            model_params_quantized = np.array(es.current_param()).round(4)
            reward_eval = evaluate_batch(model_params_quantized, max_len=-1)
            model_params_quantized = model_params_quantized.tolist()
            improvement = reward_eval - best_reward_eval
            eval_log.append([t, reward_eval, model_params_quantized])
            with open(filename_log, 'wt') as out:
                res = json.dump(eval_log, out)
            if (len(eval_log) == 1 or reward_eval > best_reward_eval):
                best_reward_eval = reward_eval
                best_model_params_eval = model_params_quantized
            else:
                if retrain_mode:
                    sprint("reset to previous best params, where best_reward_eval =", best_reward_eval)
                    es.set_mu(best_model_params_eval)
            with open(filename_best, 'wt') as out:
                res = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0, separators=(',', ': '))
            sprint("improvement", t, improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval)


def main(args):
    global scenario, optimizer, num_episode, eval_steps, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode, cap_time_mode
    scenario = args.scenario
    optimizer = args.optimizer
    num_episode = args.num_episode
    eval_steps = args.eval_steps
    num_worker = args.num_worker
    num_worker_trial = args.num_worker_trial
    antithetic = (args.antithetic == 1)
    retrain_mode = (args.retrain == 1)
    cap_time_mode= (args.cap_time == 1)
    seed_start = args.seed_start

    initialize_settings(vars(args), args.sigma_init, args.sigma_decay)

    sprint("process", rank, "out of total ", comm.Get_size(), "started")
    if (rank == 0):
        master()
    else:
        slave()

def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    (from https://github.com/garymcintire/mpi_util/)
    """
    if n<=1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # runs the given number of processes. 
        print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
        subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
        return "parent"
    else:
        global nworkers, rank
        nworkers = comm.Get_size()
        rank = comm.Get_rank()
        print('assigning the rank and nworkers', nworkers, rank)
        return "child"

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment using pepg, ses, openes, ga, cma'))
    parser.add_argument('--scenario', type=str, default='honest_basic', help='environment to train in. ')
    parser.add_argument('-o', '--optimizer', type=str, help='ses, pepg, openes, ga, cma.', default='cma')
    
    parser.add_argument('--eval_steps', type=int, default=25, help='evaluate every eval_steps step')

    # NUMBER OF CORES
    parser.add_argument('-n', '--num_worker', type=int, default=4)
    parser.add_argument('--temperature', type=float, help='temp to use', default=1.0)

    # TRIALS PER EVO STEP
    parser.add_argument('-e', '--num_episode', type=int, default=50, help='num episodes per trial')
    parser.add_argument('-t', '--num_worker_trial', type=int, help='trials per worker (how many sets of parameters each worker tries', default=32)
    
    parser.add_argument('--antithetic', type=int, default=0, help='set to 0 to disable antithetic sampling')
    parser.add_argument('--cap_time', type=int, default=0, help='set to 0 to disable capping timesteps to 2x of average.')
    parser.add_argument('--retrain', type=int, default=0, help='set to 0 to disable retraining every eval_steps if results suck.\n only works w/ ses, openes, pepg.')
    parser.add_argument('-s', '--seed_start', type=int, default=111, help='initial seed')
    parser.add_argument('--sigma_init', type=float, default=0.10, help='sigma_init')
    parser.add_argument('--sigma_decay', type=float, default=0.999, help='sigma_decay')

    parser.add_argument("--num_agents", type=int, action='store', default = 3, help='overall number of agents in simulation')

    parser.add_argument("--commit_vals", action ='store', type=str, default = ['(0,1)'], nargs='+', help="Commit values. -commit_vals (0,1) (2,0)")
    parser.add_argument("--max_round_len", action ='store', type=int, default = 32, help="")
    parser.add_argument("--null_message_val", type=int, action='store', default = 2, help='')
    
    parser.add_argument("--send_all_first_round_reward", action ='store', type=float, default = 0.3)
    parser.add_argument("--no_send_all_first_round_penalty", action ='store', type=float, default = -3.0)
    parser.add_argument("--consistency_violation", action ='store', type=float, default = -3.0, help='from the perspective of the honest. The inverse is applied to the Byzantine')
    parser.add_argument("--validity_violation", action ='store', type=float, default = -3.0)
    parser.add_argument("--majority_violation", action ='store', type=float, default = -5.0)
    parser.add_argument("--correct_commit", action ='store', type=float, default = 1.0)
    parser.add_argument("--incorrect_commit", action ='store', type=float, default = -3.0)
    parser.add_argument("--additional_round_penalty", action ='store', type=float, default = -0.5)
    parser.add_argument("--termination_penalty", action ='store', type=float, default = -5.0)
    parser.add_argument("--send_majority_value_reward", action ='store', type=float, default = .6)
    parser.add_argument("--send_incorrect_majority_value_penalty", action ='store', type=float, default = -.3)
    # Sync BA Rewards
    parser.add_argument("--first_round_reward", action ='store', type=float, default = 0)
    parser.add_argument("--PKI_penalty", action ='store', type=float, default = -1)
    parser.add_argument("--PKI_reward", action ='store', type=float, default = .25)

    
    
    #parser.add_argument("--num_byzantine", type=int, action='store', default = 1, help='overall number of byzantine agents in simulation')
    #parser.add_argument("--sample_k_size", action ='store', type=float, default = [2])


    args = parser.parse_args()
    args.commit_vals = buildTuple(args.commit_vals)
    args.commit_vals = [0,1]

    
    if "parent" == mpi_fork(args.num_worker+1): os.exit()
    main(args)



