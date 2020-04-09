from multi_agent_env.scenario import BaseScenario
from multi_agent_env.core import World, Honest_Agent


class Scenario(BaseScenario):

    def make_world(self, params):
        world = World()
        num_agents = params['num_agents']


        give_inits = list(np.random.choice([0,1], self.params['num_agents']))
        world.agents = [Honest_Agent(params, self.honest_policy, self.honest_v_function, i, give_inits) for i in range(num_agents)]

        return world


    def reset_world(self, params, world, honest_policy, value_function):
        give_inits = list(np.random.choice([0,1], self.params['num_agents']))
        world.agents = [Honest_Agent(params, honest_policy, honest_v_function, i, give_inits) for i in range(num_agents)]
    
    def benchmark_data(self, agent, world):
        #Create this method later
        pass

    # Return all honest agents
    def get_agents(self, world):
        return [agent for agent in world.agents]

    def reward(self, params, curr_sim_len, world):
        sim_done = False
        all_committed = True
        comm_values = []
        starting_values = []
        reward_list = []
        for agent in world.agents:
            if type(h.committed_value) is not int:
                all_committed = False
                break
            else: 
                comm_values.append(h.committed_value)
                starting_values.append(h.initVal)

        if all_committed: 
            sim_done = True
            honest_comm_reward , satisfied_constraints = getCommReward(params, comm_values, starting_values)
            for i, a in enumerate(agent_list):
                a.reward += honest_comm_reward

        for i, a in enumerate(world.agents):
            if a.isByzantine == False and curr_s curr_sim_len == 1 and 'send_to_all-' in a.actionStr:
                a.reward += params['send_all_first_round_reward']
        
                # round length penalties. dont incur if the agent has committed though. 
                if type(a.committed_value) is bool and not a.isByzantine and curr_sim_len == params['max_round_len']:
                    a.reward += params['termination_penalty']
                elif type(a.committed_value) is bool and not a.isByzantine:
                    a.reward += params['additional_round_penalty']
        if curr_sim_len == params['max_round_len']:
            sim_done = True

        #TODO: need to change this to a reward
        for i, agent in enumerate(world.agents):
            reward_list.append(agent.reward)
        return sim_done, reward_list

    def getCommReward(params, comm_values, starting_values):

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
            majority_init_value = np.floor((sum(starting_values)/len(starting_values))+0.5)
            if comm_values[0] != int(majority_init_value): # as already made sure they were all the same value. 
                return params['majority_violation'], satisfied_constraints

        satisfied_constraints=True
        return params['correct_commit'], satisfied_constraints

    def observation(self, agent, world):
        return agent.state

    def is_done(self, agent):
        #Not sure if this is a float or not
        return not(type(agent.committed_value) is not int)


    

    




    