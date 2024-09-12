import env.graph_traffic_env as custom

# Parallel sample action test
parallel_env = custom.parallel_env(render_mode = 'human')
observations, infos = parallel_env.reset(seed=42)

while parallel_env.agents:
    # this is where you would insert your policy
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}

    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)

parallel_env.close()