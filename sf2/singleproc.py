import retro

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import A2C


   
env = retro.make("SonicTheHedgehog-Genesis" , state="GreenHillZone.Act1")
env = DummyVecEnv([lambda: env])


model = A2C(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=1000)
model.save("sonic-a2c")

model = A2C.load("sonic-a2c")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    # print(action)
    obs, rewards, dones, info = env.step(action)
    # print(rewards)
    env.render()


