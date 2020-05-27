from multiagent.scenario import BaseScenario
from multiagent.core import World, Honest_Agent, Byzantine_Agent
import numpy as np
import torch


class Scenario(BaseScenario):

    def make_world(self, params):
        world = World(params)
        
        world.agents, world.honest_agents, world.byzantine_agents, world.one_value = self.setup_world(params)
        world.majorityValue = self.getMajority(world.agents)
        return world


    def reset_world(self, params, world):
        world.agents, world.honest_agents, world.byzantine_agents, world.one_value = self.setup_world(params)
        world.majorityValue = self.getMajority(world.agents)

    def setup_world(self, params):
        byzantine_inds = np.random.choice(range(params['num_agents']), size=params['num_byzantine'] ,replace=False)

        give_inits = list(np.random.choice([0,1], params['num_agents']))

        one_value = False
        if len(set(give_inits)) is 1:
            one_value = True

        honest_agents = []
        byzantine_agents = []
        byzantine_state = [give_inits[byzantine_inds[0]]]
        agents = []
        for i in range(params['num_agents']):
            if i in byzantine_inds:
                byzantine_agents.append(Byzantine_Agent(params, i, give_inits, byzantine_inds))
            else:
                byzantine_state.append(int(give_inits[i]))
                honest_agents.append(Honest_Agent(params, i, give_inits))
                
        # Append agents to global agent list
        agents.extend(honest_agents)
        agents.extend(byzantine_agents)

        # Initialize byzantine to have the global state
        byzantine_agents[0].state = torch.tensor(byzantine_state).int()
        # print("byzantine id: ", byzantine_agents[0].agentId)
        # print("byzantine state: ", byzantine_agents[0].state)
        # print("give_inits: ", give_inits)
        return agents, honest_agents, byzantine_agents, one_value

    def benchmark_data(self, agent, world):
        #Create this method later
        pass

    # Return all honest agents
    def get_agents(self, world):
        # TODO: why is this needed? is world agents not already a list? 
        return [agent for agent in world.agents]
    
    def reward(self, params, curr_sim_len, world):
        sim_done = False
        all_committed = True
        comm_values = []
        starting_values = []
        reward_list = []
        for agent in world.honest_agents:
            if type(agent.committed_value) is not int:
                all_committed=False
            # Check commit values
            if type(agent.committed_value) is int:
                comm_values.append(agent.committed_value)
                if agent.committed_value == world.majorityValue:
                    agent.reward += params['correct_commit']
                else:
                    agent.reward += params['majority_violation']
            ## Check not committing in first round
            if curr_sim_len == 1:
                if 'commit' in agent.actionString:
                    agent.reward += params['no_send_all_first_round_penalty']
                if 'send_to_all-new-value_' in agent.actionString:
                    sendVal = int(agent.actionString.split('_')[-1])
                    if sendVal != agent.initVal:
                        agent.reward += params['no_send_all_first_round_penalty']
            # Penalties for not committing yet
            if  type(agent.committed_value) is bool:
                if curr_sim_len < params['max_round_len']:
                    agent.reward += params['additional_round_penalty']
                elif curr_sim_len == params['max_round_len']:
                    agent.reward += params['termination_penalty']
            
        if all_committed or curr_sim_len == params['max_round_len']:
            sim_done = True

        for agent in world.byzantine_agents:
            if all_committed:
                if len(set(comm_values)) == 1 and comm_values[0] == world.majorityValue:
                    agent.reward += params['honest_correct_commit']
                else:
                    agent.reward += params['honest_incorrect_commit']
            else:
                majority_values = []
                sent_values = True
                for agent in world.honest_agents:
                    majority_values.append(self.getAgentMajority(agent))
                if sent_values:
                    if len(set(majority_values)) == 1:
                        agent.reward += params['create_conflicting_state']
                    else:
                        agent.reward += params['no_conflicting_state']
        
        for agent in world.agents:
            reward_list.append(agent.reward)

        return sim_done, reward_list


    def getCommReward(self, params, comm_values, starting_values, majorityValue):

        satisfied_constraints = False

        if len(set(comm_values)) !=1:
            return params['consistency_violation'], satisfied_constraints

        # checking validity
        if len(set(starting_values)) ==1:
            # if they are all the same and they havent 
            # agreed on the same value, then return -1
            if starting_values != comm_values:   
                return params['validity_violation'], satisfied_constraints

        # want them to commit to the majority init value: 
        if params['num_byzantine']==0:
            if comm_values[0] != majorityValue: # as already made sure they were all the same value. 
                return params['majority_violation'], satisfied_constraints

        satisfied_constraints=True
        return params['correct_commit'], satisfied_constraints

    def observation(self, agent, world):
        return agent.state

    def is_done(self, agent):
        #Not sure if this is a float or not
        return not(type(agent.committed_value) is not int)

    def getMajority(self, agents):
        starting_values = []
        for agent in agents:
            starting_values.append(agent.initVal)
        majority_init_value = np.floor((sum(starting_values)/len(starting_values))+0.5)
        return int(majority_init_value)
    
    def getAgentMajority(self, agent):
        majority_init_value = np.floor((sum(agent.state)/len(agent.state))+0.5)
        return int(majority_init_value)





    

    




