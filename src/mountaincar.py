# implementing https://blog.paperspace.com/getting-started-with-openai-gym/

import gym
env = gym.make('MountainCar-v0')

# reset the environment and see the initial observation
obs = env.reset()
print("The initial observation is {}".format(obs))

done = False

while not done:

    # Sample a random action from the entire action space
    random_action = env.action_space.sample()

    # # Take the action and get the new observation space
    new_obs, reward, done, info = env.step(random_action)
    print("The new observation is {}".format(new_obs))

    env.render(mode = "human")


env.close()
