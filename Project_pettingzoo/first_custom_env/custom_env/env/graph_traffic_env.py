import os
import math
import functools
import networkx as nx
import matplotlib.pyplot as plt
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.conversions import parallel_wrapper_fn

max_cycles = 11 #for test fairness, set high if all algos have no.steps loop

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = TrafficLightEnv(render_mode=render_mode)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)

class TrafficLightEnv(AECEnv):

    metadata = {"render_modes": ["human", "graph"], 
                "is_parallelizable": "True", 
                "graph_structures": ["lattice"]}
    
    def __init__(self, render_mode=None):
        agent_number = 36
        self.graph = self._generate_graph(agent_number)
        self.full_observability = False
        self.depth = 1  # depth for partial observability
        self.possible_agents = [str(agent) for agent in self.graph.nodes]  # Change agent identifiers to strings
        self.action_spaces = {agent: spaces.Discrete(2) for agent in self.possible_agents}  # Action space: {0: 'stop', 1: 'go'}
        
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.render_mode = render_mode
        self.reset()
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if self.full_observability:
            return spaces.MultiDiscrete([3] * len(self.graph.nodes))
        else:
            observed_nodes = self._get_observed_nodes(agent)
            #print("Agent observed nodes: ", len(observed_nodes))
            return spaces.MultiDiscrete([3] * len(observed_nodes))
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        self.agents = [str(agent) for agent in self.graph.nodes]  # Ensure agent identifiers are strings
        self.states = {str(node): 'red' for node in self.graph.nodes}  # Initialize traffic light states
        self.traffic_flows = {edge: 0 for edge in self.graph.edges}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.num_moves = 0

        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()

        return self.observations, self.infos

    def _generate_graph(self, num_agents):
        if math.sqrt(num_agents).is_integer():
            side_length = int(np.sqrt(num_agents))

            # Create a grid graph with the specified side length
            graph = nx.grid_2d_graph(side_length, side_length)

            # Relabel the nodes to be consecutive integers
            mapping = {node: i for i, node in enumerate(graph.nodes())}
            graph = nx.relabel_nodes(graph, mapping)

            return graph            
        
        else:
            graph = nx.random_geometric_graph(num_agents, radius=0.3)
            ## random graph created if grid not possible

            while not nx.is_connected(graph):
                components = list(nx.connected_components(graph))
                for i in range(len(components) - 1):
                    u = list(components[i])[0]
                    v = list(components[i + 1])[0]
                    graph.add_edge(u,v)
            
            return graph

    def observe(self, agent):
        if self.full_observability:
            return self._full_observability()
        else:
            return self._partial_observability(agent)

    def _full_observability(self):
        return np.array([self._state_to_int(self.states[str(node)]) for node in self.graph.nodes], dtype=np.int32)

    def _partial_observability(self, agent):
        observed_nodes = self._get_observed_nodes(agent)
        return np.array([self._state_to_int(self.states[str(node)]) for node in observed_nodes], dtype=np.int32)

    def _get_observed_nodes(self, agent):
        agent = int(agent)
        lengths = nx.single_source_shortest_path_length(self.graph, agent, cutoff=self.depth)
        return [node for node in lengths]

    def _state_to_int(self, state):
        return {'red': 0, 'amber': 1, 'green': 2}[state]

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return  

        agent = self.agent_selection

        self._cumulative_rewards[agent] = 0

        current_state = self.states[agent]
        new_state = self._get_new_state(current_state, action)
        self.states[agent] = new_state
        
        # Update traffic flows based on the new states
        for u, v in self.graph.edges:
            if self.states[str(u)] == 'red' and self.states[str(v)] == 'red':
                self.traffic_flows[(u, v)] = min(self.traffic_flows[(u, v)] + 2, 20)
            elif self.states[str(u)] == 'green' and self.states[str(v)] == 'green':
                self.traffic_flows[(u, v)] = max(self.traffic_flows[(u, v)] - 2, 0)
        
        self.observations[agent] = self.observe(agent)
        #print("Obs vec length: ", len(self.observations[agent]))

        if self.agent_selector.is_last():
            self.rewards = self._update_rewards()
            # print("Agent rewards", self.rewards)
            self.num_moves += 1
            self.truncations = {agent: self.num_moves >= max_cycles for agent in self.agents}
            self._accumulate_rewards()
        else:
            self._clear_rewards()
        
        self.agent_selection = self.agent_selector.next()

        if self.render_mode == 'human':            
            self.render()
        elif self.render_mode == 'graph':
            self.render_graph()
            
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _get_new_state(self, current_state, action):
        if action == 1:  # go
            if current_state == 'red':
                return 'amber'
            elif current_state == 'amber':
                return 'green'
            else:
                return current_state
        elif action == 0:  # stop
            if current_state == 'green':
                return 'amber'
            elif current_state == 'amber':
                return 'red'
            else:
                return current_state

    def _update_rewards(self):
        reward = {agent: 0 for agent in self.agents}
        for u, v in self.graph.edges:
            if self.traffic_flows[(u, v)] > 10:
                reward[str(u)] += 1
                reward[str(v)] += 1
            elif self.traffic_flows[(u, v)] <= 10 and self.traffic_flows[(u, v)] > 3:
                reward[str(u)] += 3
                reward[str(v)] += 3
            elif self.traffic_flows[(u, v)] <= 3:
                reward[str(u)] += 5
                reward[str(v)] += 5
        return reward

    def render(self):
        for node in self.graph.nodes:
            print(f"Light {node}: {self.states[str(node)]}")
        print("Traffic Flows:")
        for edge in self.graph.edges:
            print(f"From {edge[0]} to {edge[1]}: {self.traffic_flows[edge]}")
    
    def render_graph(self):
        folder_path = os.path.join(os.getcwd(), 'rendered_graphs')
        image_path = os.path.join(folder_path, 'traffic_light_graphbiglattice.png')
        
        # Check if the image already exists
        if os.path.exists(image_path):
            # print(f"Graph already exists at: {image_path}. Skipping save.")
            return
        
        G = self.graph
        pos = nx.spring_layout(G)
        cent = nx.degree_centrality(G)
        node_size = list(map(lambda x: x * 500, cent.values()))

        plt.figure(figsize=(10, 10))
        nx.draw_networkx(G, pos, width=0.25, alpha=0.3, node_size=node_size)

        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Save the image in the folder
        plt.savefig(image_path)
        plt.close()  # Close the figure to avoid displaying it in non-interactive environments
        
        print(f"Graph saved at: {image_path}")

    def close(self):
        '''
        Should close any graphical displays or anything that is opened externally to the env values. 
        Will need to close any displays or plots - so any rendered graphs must be closed
        '''
        plt.close() ##just in case - it is at end of render_graph too
        pass
