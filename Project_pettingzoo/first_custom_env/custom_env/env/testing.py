import graph_traffic_env as custom

from pettingzoo.test import parallel_api_test

env = custom.parallel_env()
parallel_api_test(env, num_cycles=1000)
