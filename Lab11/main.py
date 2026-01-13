import argparse
import time
import random
from collections import deque

import numpy as np
import cv2
import gymnasium as gym
import flappy_bird_gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def preprocess_frame(frame_rgb: np.ndarray, size=(84, 84)) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)

def init_stack(first_gray: np.ndarray, k=4) -> deque:
    st = deque(maxlen=k)
    for _ in range(k):
        st.append(first_gray)
    return st

def stack_to_state(st: deque) -> np.ndarray:
    return np.stack(list(st), axis=0)  # (4,84,84) uint8

def make_replay_buffer(capacity: int):
    return deque(maxlen=capacity)

def rb_push(rb: deque, s, a, r, ns, done):
    rb.append((s, a, r, ns, done))

def rb_sample(rb: deque, batch_size: int):
    batch = random.sample(rb, batch_size)
    s, a, r, ns, d = zip(*batch)
    s = np.array(s, dtype=np.uint8)
    ns = np.array(ns, dtype=np.uint8)
    a = np.array(a, dtype=np.int64)
    r = np.array(r, dtype=np.float32)
    d = np.array(d, dtype=np.float32)
    return s, a, r, ns, d

#DQN
def build_q_network(num_actions: int) -> nn.Module:
    #(4,84,84)
    model = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, num_actions),
    )
    return model

@torch.no_grad()
def select_action(q_net: nn.Module, state_u8: np.ndarray, epsilon: float, num_actions: int, device: str):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    x = torch.from_numpy(state_u8).unsqueeze(0).to(device)  # (1,4,84,84)
    q = q_net(x.float() / 255.0)
    return int(torch.argmax(q, dim=1).item())

def huber_loss(q_pred, q_target):
    return F.smooth_l1_loss(q_pred, q_target)

def dqn_update_step(q_net, target_net, optimizer, batch, gamma: float, device: str):
    s, a, r, ns, d = batch
    s  = torch.from_numpy(s).to(device).float() / 255.0  #(B,4,84,84)
    ns = torch.from_numpy(ns).to(device).float() / 255.0
    a  = torch.from_numpy(a).to(device) # (B,)
    r  = torch.from_numpy(r).to(device) # (B,)
    d  = torch.from_numpy(d).to(device) # (B,)

    q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        #Double DQN
        next_actions = q_net(ns).argmax(dim=1) #optimal action
        next_q = target_net(ns).gather(1, next_actions.unsqueeze(1)).squeeze(1)  #evalute with target net
        target = r + gamma * (1.0 - d) * next_q

    loss = huber_loss(q_sa, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()

    return float(loss.item())

def linear_epsilon(step, eps_start, eps_end, eps_decay_steps):
    if step >= eps_decay_steps:
        return eps_end
    t = step / eps_decay_steps
    return eps_start + t * (eps_end - eps_start)

def get_config():
    cfg = {}

    # training
    cfg["total_steps"] = 200000
    cfg["gamma"] = 0.99
    cfg["lr"] = 1e-4
    cfg["batch_size"] = 32
    cfg["warmup_steps"] = 5000

    cfg["replay_capacity"] = 100000

    cfg["eps_start"] = 1.0
    cfg["eps_end"] = 0.05
    cfg["eps_decay_steps"] = 100000

    #DQN
    cfg["target_update_every"] = 2000
    cfg["train_every"] = 4

    #pixel procesing
    cfg["stack_k"] = 4
    cfg["frame_skip"] = 2

    cfg["print_every"] = 10

    # paths
    cfg["model_path"] = "dqn_flappy_pixels.pt"

    return cfg

@torch.no_grad()
def evaluate(q_net, episodes, stack_k, frame_skip, device):
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    q_net.eval()
    scores = []
    for _ in range(episodes):
        obs, info = env.reset()
        frame = env.render()
        g0 = preprocess_frame(frame)
        st = init_stack(g0, k=stack_k)
        state = stack_to_state(st)

        ep_r = 0.0
        done = False
        while not done:
            x = torch.from_numpy(state).unsqueeze(0).to(device).float() / 255.0
            action = int(torch.argmax(q_net(x), dim=1).item())

            total_r = 0.0
            for _ in range(frame_skip):
                obs, r, terminated, truncated, info = env.step(action)
                total_r += float(r)
                frame = env.render()
                g = preprocess_frame(frame)
                st.append(g)
                state = stack_to_state(st)
                done = terminated or truncated
                if done:
                    break
            ep_r += total_r

        scores.append(ep_r)

    env.close()
    q_net.train()
    return float(np.mean(scores)), float(np.std(scores))

#TRAINING
def train(cfg):
    device = get_device()
    best_eval = -1e9
    print("Device:", device)
    eval_steps, eval_means, eval_stds = [], [], []

    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    num_actions = env.action_space.n

    q_net = build_q_network(num_actions).to(device)
    target_net = build_q_network(num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg["lr"])

    rb = make_replay_buffer(cfg["replay_capacity"])

    episode_rewards = []
    losses = []

    obs, info = env.reset()
    frame = env.render()
    g0 = preprocess_frame(frame)
    st = init_stack(g0, k=cfg["stack_k"])
    state = stack_to_state(st)

    ep_reward = 0.0
    episode_idx = 0
    global_step = 0

    while global_step < cfg["total_steps"]:
        eps = linear_epsilon(
            global_step,
            cfg["eps_start"],
            cfg["eps_end"],
            cfg["eps_decay_steps"]
        )

        action = select_action(q_net, state, eps, num_actions, device)

        total_r = 0.0
        done = False
        for _ in range(cfg["frame_skip"]):
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += float(reward)

            frame = env.render()
            g = preprocess_frame(frame)
            st.append(g)
            next_state = stack_to_state(st)

            done = terminated or truncated
            if done:
                break

        rb_push(rb, state, action, total_r, next_state, float(done))
        state = next_state
        ep_reward += total_r
        global_step += 1

        if len(rb) >= cfg["warmup_steps"] and global_step % cfg["train_every"] == 0:
            batch = rb_sample(rb, cfg["batch_size"])
            loss = dqn_update_step(
                q_net, target_net, optimizer, batch, cfg["gamma"], device
            )
            losses.append(loss)

        if global_step % cfg["target_update_every"] == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            episode_rewards.append(ep_reward)
            episode_idx += 1

            if episode_idx % cfg["print_every"] == 0:
                avg = np.mean(episode_rewards[-cfg["print_every"]:])
                print(
                    f"Ep {episode_idx:4d} | step {global_step:6d} | "
                    f"R {ep_reward:7.2f} | avg {avg:7.2f} | eps {eps:5.2f}"
                )

            obs, info = env.reset()
            frame = env.render()
            g0 = preprocess_frame(frame)
            st = init_stack(g0, k=cfg["stack_k"])
            state = stack_to_state(st)
            ep_reward = 0.0

        if global_step % 10000 == 0 and global_step > 0:
            m, s = evaluate(q_net, episodes=5, stack_k=cfg["stack_k"],
                            frame_skip=cfg["frame_skip"], device=device)
            eval_steps.append(global_step)
            eval_means.append(m)
            eval_stds.append(s)
            print(f" Eval step {global_step} | mean {m:.2f} ± {s:.2f}")

            if m > best_eval:
                best_eval = m
                torch.save(q_net.state_dict(), f"dqn_best_step_{global_step}.pt")
                print(f"Best saved dqn_best.pt | best mean {best_eval:.2f}")

    torch.save(q_net.state_dict(), cfg["model_path"])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps", type=int)
    p.add_argument("--replay_capacity", type=int)
    p.add_argument("--warmup_steps", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--gamma", type=float)
    p.add_argument("--lr", type=float)

    p.add_argument("--eps_start", type=float)
    p.add_argument("--eps_end", type=float)
    p.add_argument("--eps_decay_steps", type=int)

    p.add_argument("--target_update_every", type=int)
    p.add_argument("--train_every", type=int)

    p.add_argument("--stack_k", type=int)
    p.add_argument("--frame_skip", type=int)

    p.add_argument("--print_every", type=int)

    p.add_argument("--model_path", type=str, default="dqn_flappy_pixels.pt")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
