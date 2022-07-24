import gym
import time
import matplotlib.pyplot as plt
from breakout.action_wrapper import ActionWrapper
from breakout.concat_obs import ConcatObs
from breakout.observation_wrapper import ObservationWrapper
from breakout.reward_wrapper import RewardWrapper 

env = gym.make("BreakoutNoFrameskip-v4", render_mode='human')
env.metadata['render_fps'] = 30

print('actions', env.unwrapped.get_action_meanings())


wrapped_env = ObservationWrapper(RewardWrapper(ActionWrapper(ConcatObs(env, 4))))
print("The new observation space is", wrapped_env.observation_space)

print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)


obs = wrapped_env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = wrapped_env.step(action)
    # Raise a flag if values have not been vectorised properly
    if (obs > 1.0).any() or (obs < 0.0).any():
        print("Max and min value of observations out of range")
    
    # Raise a flag if reward has not been clipped.
    if reward < 0.0 or reward > 1.0:
        assert False, "Reward out of bounds"
    print('reward', reward)
    image = env.render('rgb_array')
    plt.imshow(image)
    time.sleep(0.01)
wrapped_env.close()
print("All checks passed")

