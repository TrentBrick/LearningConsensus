#from config import *

if __name__=='__main__':
    from environment_utils import initStatesandAgents

    agent_list = initStatesandAgents()
    for i in range(3):
        print(i)
        print(agent_list[i].actionSpace)
        print(agent_list[i].initState)
        print(agent_list[i].agentID)
        print(agent_list[i].isByzantine)