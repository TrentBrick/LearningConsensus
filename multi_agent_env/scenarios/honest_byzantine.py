from multi_agent_env.scenario import BaseScenario


###TODO: Need to finish this environment, this is just boilerplate code
class Scenario(BaseScenario):

    def make_world(self, params):
        world = World()
        num_agents = params['num_agents']

    def reset_world(self, params):
    
    def benchmark_data(self, agent, world):
        #Create this method later
        pass

    # Return all honest agents
    def get_honest_agents(self, world):
        return [agent for agent in world.agents if not agent.isByzantine]

    # Return all byzantine agents
    def get_byzantine_agents(self, world):
        return [agent for agent in world.agents if  agent.isByzantine]

###