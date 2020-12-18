#from config import *
import itertools

if __name__=='__main__':
    # from environment_utils import initStatesandAgents

    # agent_list = initStatesandAgents()
    # for i in range(3):
    #     print(i)
    #     print(agent_list[i].actionSpace)
    #     print(agent_list[i].initState)
    #     print(agent_list[i].agentID)
    #     print(agent_list[i].isByzantine)
    agent_state = [0,1,2,3,4,5,6]
    other_state = [4,5,6,7,8,9,10]
    cartesian_prod = itertools.product(agent_state[0:5], other_state[0:5])

    cartesian_list = list(cartesian_prod)
    print("prod is: ", cartesian_list)

    final_state = []
    for pair in cartesian_list:
        final_state.append(pair[0])
        final_state.append(pair[1])

    print("final state is: ", final_state)
    print("len is: ", len(final_state))

    new_state = []
    new_state = new_state + [0,1]*50
    print('new_state is: ', new_state)
    print("len is: ", len(new_state))
    