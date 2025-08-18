import yaml
import torch
import numpy as np
from env.crowd_env import CrowdNavEnv
from models.camrl_agent import CAMRLAgent
from tqdm import trange

def evaluate_il(
    cfg_path="config/default.yaml",
    encoder_path="weights/camrl_final_encoder.pth",
    policy_head_path="weights/camrl_final_policy_head.pth",
    eval_episodes=100
):
    cfg = yaml.safe_load(open(cfg_path))
    mamba_cfg = cfg['mamba']

    env = CrowdNavEnv(cfg)
    agent = CAMRLAgent(cfg['obs_dim'], mamba_cfg['state_dim'], cfg)

    agent.encoder.load_state_dict(torch.load(encoder_path, map_location=agent.device))
    agent.policy_head.load_state_dict(torch.load(policy_head_path, map_location=agent.device))
    agent.encoder.eval()
    agent.policy_head.eval()

    stats = {"reach_goal": 0, "collision": 0, "timeout": 0, "other": 0}
    total_steps = []

    for ep in trange(eval_episodes, desc="Evaluate IL Episodes", ncols=80):
        state_seq = env.reset()
        done = False
        steps = 0
        last_event = "other"
        while not done:
            # 只用policy_head决策
            state_seq_tensor = torch.from_numpy(np.array(state_seq, dtype=np.float32)).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                h = agent.encoder(state_seq_tensor)
                action = agent.policy_head(h).cpu().numpy().flatten()
            state_seq, reward, done, info = env.step(action)
            steps += 1
            if done and "event" in info:
                last_event = info["event"]
        stats[last_event] += 1
        total_steps.append(steps)

    print(f"====== IL-only Evaluation Summary ({eval_episodes} episodes) ======")
    for k, v in stats.items():
        print(f"{k:12s}: {v}  ({v/eval_episodes*100:.1f}%)")
    print(f"Average Steps per Episode: {np.mean(total_steps):.2f}")

if __name__ == '__main__':
    evaluate_il()

