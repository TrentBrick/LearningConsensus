actionMap = {
    0: (0, getCommitVals),
    1: (0, getSendAnyCommitVal),
    2: (0, getSendInitval),
    3: (0, getSampleAvalancheBasic),
    4: (1, getFullByzantineActionSpace) ##Need to pass in the byzantine indexes - set value to 1
}

# *****
    # actionMap: Dictionary mapping action index to a tuple(needs byzantine index, function)

    # Creating a new function: 
    #   1. Increment key to actionMap and give it a value with a tuple and functionName
            ## If action is byzantine and needs byzantine indexes, first value of tuple is 1
            ## Else, first value of tuple is 0
    #   2. Create function below getActionSpace

    # Running with specified Actions: 
    #   1. When running with actions correspondign to keys 1,2,3 - do run.py --actions[1,2,3]

# *****



def getActionSpace(params, isByzantine, byzantine_inds=None):
    action_list = params['action_list']
    action_space = []
    for action in action_list:
        ## If action is byzantine - pass in byzantine indexes
        if (actionMap[action][0] and isByzantine):
            curr_action_actions = actionMap[action](params, byzantine_inds)
        ## Don't pass in byzantine indexes
        else:
            curr_action_actions = actionMap[action](params)
        for single_action in curr_action_actions:
            action_space.append(single_action)

    return action_space

def getCommitValues(params):
    action_space = []
    for commit_val in params['commit_vals']:
        action_space.append('commit_'+str(commit_val))
    return action_space

def getSendAnyCommitVal(params):
    action_space = []
    for commit_val in params['commit_vals']:
        action_space.append('send_to_all-value_'+str(commit_val))
    return action_space

def getSendInitVal(params):
    action_space = ['send_to_all-value_init']
    return action_space

def getSampleAvalancheBasic(params):
    action_space = ['sample_k_avalanche']
    return action_space

def getFullByzantineActionSpace(params, byzantine_inds):
    action_space = []
    
    action_space.append('no_send')
    # no point in sending messages to other Byzantines as the central agent knows what the states are
    # but do have v granular send options.
    # and no commit option
    # get every possible combination of sending actions possible

    # remove the byz agents.
    non_byzantines = list(range(0, params['num_agents']))
    #print('byzantine inds', byzantine_inds)
    #print(non_byzantines)
    for byzantine_ind in byzantine_inds:
        if byzantine_ind in non_byzantines:
            non_byzantines.remove(byzantine_ind)
    #print('non byz are', non_byzantines)
    #for val in commit_vals:
    #    non_byzantines.append('v'+str(val)) # add in the possible values that can be sent

    # this code is tricky, I get all combinations of the honest agents to send to
    # and then interleave in all permutations of the values that can be sent to them.
    # for example a subset of them is: :
    ''''send_agent-2_value-0',
    'send_agent-1_value-0_agent-3_value-0',
    'send_agent-1_value-0_agent-2_value-1_agent-3_value-1',
    'send_agent-1_value-1_agent-3_value-1',
    'send_agent-1_value-1_agent-2_value-1_agent-3_value-0',
    'send_agent-1_value-0_agent-2_value-0_agent-3_value-0',
    'send_agent-1_value-1_agent-2_value-1',
    'send_agent-2_value-1_agent-3_value-1','''
    for choose_n in range(1, len(non_byzantines)+1):
        commit_val_permutes = list(itertools.permutations(params['commit_vals']*((choose_n//2)+1)))
        for combo_el in itertools.combinations(non_byzantines, choose_n):
            for cvp in commit_val_permutes:
                string = 'send'
                for ind in range(choose_n):
                    string += '_agent-'+str(combo_el[ind])+'_v-'+str(cvp[ind])
                    #print('string', string)
                action_space.append( string )
    # remove any redundancies in a way that preserves order.
    action_space = list(OrderedDict.fromkeys(action_space))
    return action_space

def actionEffect(params, actionStr, init_val, actor_prev_action_result, receiver_id):
    # return the effects of a particular action

    if actionStr == 'no_send':
        return params['null_message_val']

    elif 'commit' in actionStr: # keep returning the last state that the agent sent
        return actor_prev_action_result

    elif actionStr == 'send_to_all-value_init':
        return init_val

    elif 'send_to_all-value_' in actionStr:
        return int(actionStr.split('_')[-1])
    
    elif 'agent-'+str(receiver_id) in actionStr:
        # getting the first value of this. 
        return int(actionStr.split('agent-'+str(receiver_id)+'_v-')[-1][0])

    else:
        return params['null_message_val']
