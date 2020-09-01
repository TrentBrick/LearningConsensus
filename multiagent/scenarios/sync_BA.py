from multiagent.scenario import BaseScenario
from multiagent.core_sync_BA import World, Honest_Agent, Byzantine_Agent
import numpy as np
import torch


class Scenario(BaseScenario):

    def make_world(self, params):
        world = World(params)
    
        world.agents, world.honest_agents, world.byzantine_agents = self.setup_world(params)

        world.byzantineEquivocate = False
        #Majority value is irrelevant right now - will be used later 
        world.majorityValue = -1
        return world


    def reset_world(self, params, world):
        world.agents, world.honest_agents, world.byzantine_agents = self.setup_world(params)
        # print(world.byzantine_agents[0].state)
        world.byzantineEquivocate = False
        world.majorityValue = -1

    def setup_world(self, params):
        byzantine_inds = np.random.choice(range(params['num_agents']), size=params['num_byzantine'] ,replace=False)

        give_inits = list(np.random.choice([0,1], params['num_agents']))

        one_value = False
        if len(set(give_inits)) == 1:
            one_value = True

        honest_agents = []
        byzantine_agents = []
        agents = []
        for i in range(params['num_agents']):
            if i in byzantine_inds:
                byzantine_agents.append(Byzantine_Agent(params, i, give_inits, byzantine_inds))
            else:
                honest_agents.append(Honest_Agent(params, i, give_inits))
                
        # Append agents to global agent list
        agents.extend(honest_agents)
        agents.extend(byzantine_agents)

        # Initialize byzantine to have the global state
        return agents, honest_agents, byzantine_agents

    def benchmark_data(self, agent, world):
        #Create this method later
        pass

    # Return all honest agents
    def get_agents(self, world):
        # TODO: why is this needed? is world agents not already a list? 
        return [agent for agent in world.agents]
    
    def reward(self, params, curr_sim_len, world):
        sim_done = False
        reward_list = []
        safety_violation = False
        for byz_agent in world.byzantine_agents:
            if curr_sim_len%4 == 1:
                pass
            if curr_sim_len%4 == 2:
                # byz_agent.reward += params['first_round_reward']
                if world.byzantineEquivocate:
                    byz_agent.reward += params['equivocation_penalty']
            if curr_sim_len%4 == 3:
                # pass
                if world.byzantineEquivocate:
                    byz_agent.reward += params['equivocation_penalty']
                # if byz_agent.prevActionString != byz_agent.actionString:
                #     byz_agent.reward += params['PKI_penalty']
                # else:
                #     byz_agent.reward += params['PKI_reward']
                # for agent in world.honest_agents:
            if curr_sim_len%4 == 0:
                comm_values = []
                for agent in world.honest_agents:
                    comm_values.append(agent.committed_value)
                #If one agent has committed and the other hasn't, give a reward
                if (2 in comm_values) and (1 in comm_values or 0 in comm_values):
                    byz_agent.reward += params['termination_reward']
                elif 1 in comm_values and 0 in comm_values:
                    byz_agent.reward += params['safety_reward']
                    safety_violation = True
                elif (len(set(comm_values)) is 1) and (1 in comm_values or 0 in comm_values) and (2 not in comm_values):
                    byz_agent.reward += params['honest_correct_commit']
                    sim_done = True
        
        for agent in world.byzantine_agents:
            reward_list.append(agent.reward)

        ## Code for checking if simulation is done -> done when we want to force end after x rounds ##
        # if curr_sim_len == params['max_round_len']:
        #     sim_done = True

        return sim_done, reward_list, safety_violation

    def observation(self, agent, world):
        return agent.state

    def is_done(self, agent):
        return agent.committed_value != 2

    def getMajority(self, agents):
        starting_values = []
        for agent in agents:
            starting_values.append(agent.initVal)
        majority_init_value = np.floor((sum(starting_values)/len(starting_values))+0.5)
        return int(majority_init_value)
    
    def getAgentMajority(self, agent):
        majority_init_value = np.floor((sum(agent.state)/len(agent.state))+0.5)
        return int(majority_init_value)





    




