# Behavioural Planning for Autonomous Highway Driving

# Setup
# import useful modules for the environment, agent, and visualization.

from tqdm.notebook import trange
import sys
# Agent
from rl_agents.agents.common.factory import agent_factory

# Environment
import highway_env
import gym
sys.path.insert(0, './highway-env/scripts/')

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s: %(message)s"
)
logging.info(
    "imported useful modules for the environment, agent, and visualization")

# Make environment
highway_env_type = "highway-fast-v0"
logging.info("Make environment {}".format(highway_env_type))
env = gym.make(highway_env_type)
vid = gym.wrappers.monitoring.video_recorder.VideoRecorder(
    env=env, path="{}.mp4".format(highway_env_type))
obs, done = env.reset(), False

# Make agent
logging.info("Make Agent")
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method": "simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

# Run episode
logging.info("Running Episodes...")
for step in trange(env.unwrapped.config["duration"], desc="Running..."):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    vid.capture_frame()

env.close()
logging.info("Done!!!")
vid.close()
