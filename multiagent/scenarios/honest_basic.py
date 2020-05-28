from multiagent.scenario import BaseScenario
from multiagent.core import World, Honest_Agent
import numpy as np


class Scenario(BaseScenario):

    def make_world(self, params):
        world = World(params)
        num_agents = params['num_agents']

        give_inits = list(np.random.choice([0,1], params['num_agents']))
        world.agents = [Honest_Agent(params, i, give_inits) for i in range(num_agents)]
        world.majorityValue = self.getMajority(world.agents)

        #Just placeholders to comply with environment needs
        world.honest_agents = []
        world.byzantine_agents = []
        return world

    def reset_world(self, params, world):
        give_inits = list(np.random.choice([0,1], params['num_agents']))
        world.agents = [Honest_Agent(params, i, give_inits) for i in range(params['num_agents'])]
        world.majorityValue = self.getMajority(world.agents)

        world.honest_agents = []
        world.byzantine_agents = []


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
        for agent in world.agents:
            if type(agent.committed_value) is not int:
                all_committed=False
            # Check commit values
            if type(agent.committed_value) is int:
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
            
            reward_list.append(agent.reward)
        if all_committed or curr_sim_len == params['max_round_len']:
            sim_done = True

        return sim_done, reward_list
    # def reward(self, params, curr_sim_len, world):
    #     sim_done = False
    #     all_committed = True
    #     comm_values = []
    #     starting_values = []
    #     reward_list = []
    #     for agent in world.agents:
    #         if type(agent.committed_value) is not int:
    #             all_committed=False
    #         # Check commit values
    #         if type(agent.committed_value) is int:
    #             if agent.committed_value == world.majorityValue:
    #                 agent.reward += params['correct_commit']
    #             else:
    #                 agent.reward += params['majority_violation']
    #         ## Check majority values
    #         if curr_sim_len > 1 and type(agent.committed_value) is bool:
    #         ## Check sending of majority value
    #             if 'send_to_all-new-value_' in agent.actionString and not agent.sentMajority:
    #                 if int(agent.actionString.split('_')[-1]) == world.majorityValue:
    #                     agent.reward+= params['send_majority_value_reward']
    #                     agent.sentMajority = True
    #                 else:
    #                     agent.reward+= params['send_incorrect_majority_value_penalty']
    #         ## Check
    #         if curr_sim_len == 1:
    #             if 'send_to_all-value_init' in agent.actionString:
    #                 agent.reward += params['send_all_first_round_reward']
    #             else:
    #                 agent.reward += params['no_send_all_first_round_penalty']
    #         # Penalties for not committing yet
    #         if  type(agent.committed_value) is bool:
    #             if curr_sim_len < params['max_round_len']:
    #                 agent.reward += params['additional_round_penalty']
    #             elif curr_sim_len == params['max_round_len']:
    #                 agent.reward += params['termination_penalty']
            
    #         reward_list.append(agent.reward)
    #     if all_committed or curr_sim_len == params['max_round_len']:
    #         sim_done = True

    #     return sim_done, reward_list
    # def old_reward(self, params, curr_sim_len, world):
    #     sim_done = False
    #     all_committed = True
    #     comm_values = []
    #     starting_values = []
    #     reward_list = []
    #     for agent in world.agents:
    #         if type(agent.committed_value) is not int:
    #             all_committed = False
    #         else: 
    #             comm_values.append(agent.committed_value)
    #         starting_values.append(agent.initVal)
    #     if all_committed: 
    #         sim_done = True
    #         honest_comm_reward , satisfied_constraints = self.getCommReward(params, comm_values, starting_values, world.majorityValue)
    #         for i, a in enumerate(world.agents):
    #             a.reward += honest_comm_reward

    #     if not all_committed and len(comm_values) > 0 and curr_sim_len > 1:
    #         for agent in world.agents:
    #             reward=0
    #             if 'commit' in agent.actionString:
    #                 if agent.committed_value != int(world.majorityValue): # as already made sure they were all the same value. 
    #                     reward = params['majority_violation']
    #                 else:
    #                     reward = params['correct_commit']
    #                 agent.reward+=reward

    #     for i, a in enumerate(world.agents):
    #         if a.isByzantine == False and curr_sim_len == 1 and 'send_to_all-' in a.actionString:
    #             a.reward += params['send_all_first_round_reward']
    #         # elif a.isByzantine and curr_sim_len == 1 and 'send-to-all-' not in a.actionString:
    #         #     a.reward += params['no-send_all_first_round_penalty']
    #         # round length penalties. dont incur if the agent has committed though. 
    #         if type(a.committed_value) is bool and not a.isByzantine and curr_sim_len == params['max_round_len']:
    #             a.reward += params['termination_penalty']
    #         elif type(a.committed_value) is bool and not a.isByzantine:
    #             a.reward += params['additional_round_penalty']
        
    #     if curr_sim_len == params['max_round_len']:
    #         sim_done = True

    #     #TODO: need to change this to a reward
    #     for i, agent in enumerate(world.agents):
    #         reward_list.append(agent.reward)
    #     return sim_done, reward_list

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




    

    




    