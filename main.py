'''Train a Dots and Boxes agent using PPO with PufferLib.'''

import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib
import pufferlib.vector

from dots_and_boxes import DotsAndBoxes


class Policy(nn.Module):
    def __init__(self, obs_size, act_size, hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, act_size)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.float()
        hidden = self.network(x)
        return self.actor(hidden), self.critic(hidden)

    def get_action_and_value(self, obs, action=None):
        logits, value = self(obs)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)

    def get_value(self, obs):
        _, value = self(obs)
        return value.squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=3)
    parser.add_argument('--cols', type=int, default=3)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--num-steps', type=int, default=128)
    parser.add_argument('--num-updates', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-coef', type=float, default=0.2)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--render', action='store_true', help='Live render during training')
    parser.add_argument('--render-fps', type=int, default=10, help='FPS for live render')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--save-interval', type=int, default=50, help='Save checkpoint every N updates')
    parser.add_argument('--self-play', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable self-play training (default: on)')
    parser.add_argument('--self-play-start', type=int, default=50,
                        help='Train vs random for this many updates before self-play')
    parser.add_argument('--opponent-update-interval', type=int, default=25,
                        help='Copy weights to opponent every N updates after self-play starts')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    import os
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.checkpoint_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f'Run directory: {run_dir}')

    # Create vectorized environments
    envs = pufferlib.vector.make(
        DotsAndBoxes,
        env_kwargs=dict(rows=args.rows, cols=args.cols),
        backend=pufferlib.vector.Serial,
        num_envs=args.num_envs,
    )

    # Optional live render env (separate single env for display)
    renderer = None
    render_env = None
    if args.render:
        from dots_and_boxes import RaylibRenderer
        render_env = DotsAndBoxes(rows=args.rows, cols=args.cols, render_mode='human')
        render_env.reset()
        renderer = RaylibRenderer(render_env, fps=args.render_fps)

    obs_size = envs.single_observation_space.shape[0]
    act_size = envs.single_action_space.n
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // 4

    policy = Policy(obs_size, act_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_size), device=device)
    act_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logprob_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    reward_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    value_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    global_step = 0
    start_time = time.time()
    episode_returns = []

    # Self-play opponent (shared across all envs)
    import copy
    opponent_policy = None

    print(f'Training Dots and Boxes ({args.rows}x{args.cols} boxes)')
    print(f'Obs size: {obs_size}, Action size: {act_size}')
    print(f'Num envs: {args.num_envs}, Batch size: {batch_size}')
    if args.self_play:
        print(f'Self-play: on (starts at update {args.self_play_start}, '
              f'opponent updates every {args.opponent_update_interval})')
    print()

    for update in range(1, args.num_updates + 1):
        # Anneal learning rate
        frac = 1.0 - (update - 1.0) / args.num_updates
        optimizer.param_groups[0]['lr'] = frac * args.learning_rate

        # Collect rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(next_obs)

            act_buf[step] = action
            logprob_buf[step] = logprob
            value_buf[step] = value

            next_obs_np, reward, terminals, truncations, infos = envs.step(
                action.cpu().numpy())
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            reward_buf[step] = torch.tensor(reward, device=device)
            next_done = torch.tensor(
                np.logical_or(terminals, truncations), dtype=torch.float32, device=device)

            for info in infos:
                if 'reward' in info:
                    episode_returns.append(info['reward'])

        # GAE advantage computation
        with torch.no_grad():
            next_value = policy.get_value(next_obs)
            advantages = torch.zeros_like(reward_buf)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = value_buf[t + 1]
                delta = reward_buf[t] + args.gamma * nextvalues * nextnonterminal - value_buf[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
            returns = advantages + value_buf

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_size)
        b_actions = act_buf.reshape(-1)
        b_logprobs = logprob_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_buf.reshape(-1)

        # PPO update
        b_inds = np.arange(batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8)

                # Clipped surrogate loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        # Self-play: activate or update opponent
        if args.self_play:
            if update == args.self_play_start:
                opponent_policy = copy.deepcopy(policy).eval()
                for env in envs.envs:
                    env.opponent_policy = opponent_policy
                print(f'[Self-play] Activated at update {update}')
            elif (opponent_policy is not None
                  and update > args.self_play_start
                  and (update - args.self_play_start) % args.opponent_update_interval == 0):
                opponent_policy.load_state_dict(policy.state_dict())
                print(f'[Self-play] Opponent updated at update {update}')

        # Save checkpoint
        if update % args.save_interval == 0 or update == args.num_updates:
            path = f'{run_dir}/dots_and_boxes_{update}.pt'
            torch.save(policy.state_dict(), path)

        # Live render: play a few steps with current policy in the render env
        if renderer is not None:
            if renderer.should_close():
                renderer.close()
                renderer = None
            else:
                with torch.no_grad():
                    for _ in range(5):
                        obs_t = torch.tensor(
                            render_env.observations, dtype=torch.float32, device=device)
                        logits, _ = policy(obs_t)
                        action = torch.distributions.Categorical(logits=logits).sample()
                        render_env.step(action.cpu().numpy())
                        renderer.render()

        # Logging
        if update % 10 == 0 or update == 1:
            sps = int(global_step / (time.time() - start_time))
            if episode_returns:
                recent = episode_returns[-100:]
                avg_ret = np.mean(recent)
                wins = sum(1 for r in recent if r > 0) / len(recent) * 100
                print(f'Update {update:4d} | Step {global_step:7d} | '
                      f'SPS {sps:5d} | Avg Return {avg_ret:+.2f} | '
                      f'Win Rate {wins:.0f}% | Episodes {len(episode_returns)}')
            else:
                print(f'Update {update:4d} | Step {global_step:7d} | SPS {sps:5d}')

    envs.close()
    if renderer is not None:
        renderer.close()

    # Save final checkpoint
    final_path = f'{run_dir}/dots_and_boxes_final.pt'
    torch.save(policy.state_dict(), final_path)
    print(f'\nCheckpoint saved to {final_path}')

    elapsed = time.time() - start_time
    print(f'Training complete in {elapsed:.1f}s ({global_step} total steps)')

    if episode_returns:
        last = episode_returns[-100:]
        print(f'Final avg return: {np.mean(last):+.2f}')
        print(f'Final win rate: {sum(1 for r in last if r > 0) / len(last) * 100:.0f}%')


if __name__ == '__main__':
    main()
