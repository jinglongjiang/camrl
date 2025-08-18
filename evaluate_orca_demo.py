import numpy as np
from env.crowd_env import CrowdNavEnv
import yaml
from tqdm import trange

CONFIG_FILE = 'config/default.yaml'
DEMO_FILE = 'orca_demos_seq.npz'

def evaluate_success_rate(cfg_file=CONFIG_FILE, demo_file=DEMO_FILE, n_eval=20, max_steps=300):
    # 加载环境配置
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    env = CrowdNavEnv(cfg)

    # 加载演示数据（每集独立）
    data = np.load(demo_file, allow_pickle=True)
    obs_list = data['obs']
    act_list = data['act']
    n_total = len(obs_list)
    print(f"总共有 {n_total} 集demo")

    reach_goal = 0
    collision = 0
    timeout = 0

    eval_num = min(n_eval, n_total)
    for i in trange(eval_num, desc="Evaluating Expert Demos"):
        ep_act = act_list[i]
        env_obs = env.reset()
        done = False
        step_idx = 0
        final_event = "Timeout"
        while not done and step_idx < len(ep_act) and step_idx < max_steps:
            a = ep_act[step_idx]
            env_obs, _, done, info = env.step(a)
            if "event" in info:
                final_event = info["event"]
            step_idx += 1

        if final_event.lower() in ["reach_goal", "reached_goal"]:
            reach_goal += 1
        elif final_event.lower() == "collision":
            collision += 1
        else:
            timeout += 1

    print("\n评测完毕：")
    print(f"总测试条数：{eval_num}")
    print(f"到达目标：{reach_goal}，占比 {reach_goal / eval_num:.2%}")
    print(f"发生碰撞：{collision}，占比 {collision / eval_num:.2%}")
    print(f"超时或其他：{timeout}，占比 {timeout / eval_num:.2%}")

if __name__ == "__main__":
    evaluate_success_rate()

