import time
import cv2
import numpy as np
import torch
import gymnasium as gym
import flappy_bird_gymnasium

from main import get_device, build_q_network, preprocess_frame, init_stack, stack_to_state

@torch.no_grad()
def play_visual(model_path, stack_k=4, frame_skip=2, fps_cap=60):
    device = get_device()
    print("Device:", device)

    env = gym.make("FlappyBird-v0", render_mode="rgb_array")  # MUST be rgb_array
    num_actions = env.action_space.n

    q_net = build_q_network(num_actions).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    obs, info = env.reset()
    frame = env.render()
    if frame is None:
        raise RuntimeError("Frame is None")

    g0 = preprocess_frame(frame)
    st = init_stack(g0, k=stack_k)
    state = stack_to_state(st)

    done = False
    total_reward = 0.0
    delay_ms = int(1000 / max(1, fps_cap))

    while not done:
        #RGB -> BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("FlappyBird - Agent", bgr)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            break

        # agent action
        x = torch.from_numpy(state).unsqueeze(0).to(device).float() / 255.0
        action = int(torch.argmax(q_net(x), dim=1).item())

        # frame skip
        for _ in range(frame_skip):
            obs, r, terminated, truncated, info = env.step(action)
            total_reward += float(r)

            frame = env.render()
            if frame is None:
                raise RuntimeError("Frame None during episode.")

            g = preprocess_frame(frame)
            st.append(g)
            state = stack_to_state(st)

            done = terminated or truncated
            if done:
                break

    print("Final reward:", total_reward)
    env.close()
    cv2.destroyAllWindows()

play_visual("dqn_best_step_150000.pt")