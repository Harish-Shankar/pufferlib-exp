# Dots and Boxes

A Dots and Boxes reinforcement learning environment built with [PufferLib](https://github.com/PufferAI/PufferLib), trained with PPO, and visualized with raylib.

## Setup

```bash
uv sync
```

## Usage

### Train an agent

```bash
uv run main.py
```

Watch it learn in real-time with a raylib window:

```bash
uv run main.py --render
```

Training options:

| Flag | Default | Description |
|------|---------|-------------|
| `--rows` | 3 | Grid rows |
| `--cols` | 3 | Grid columns |
| `--num-envs` | 8 | Parallel environments |
| `--num-updates` | 500 | Training updates |
| `--render` | off | Live render during training |
| `--render-fps` | 10 | Live render speed |
| `--save-interval` | 50 | Checkpoint frequency (updates) |
| `--checkpoint-dir` | `checkpoints/` | Where to save models |

### Watch a trained agent play

```bash
uv run dots_and_boxes.py --checkpoint checkpoints/dots_and_boxes_final.pt
```

### Watch random play

```bash
uv run dots_and_boxes.py
```

### Benchmark environment speed

```bash
uv run dots_and_boxes.py --benchmark
```

## How it works

Two players take turns drawing lines between adjacent dots on a grid. Completing the 4th side of a box scores a point and grants another turn. The agent (blue) plays against a random opponent (red).

The PPO agent learns to:
- Claim boxes when three sides are drawn
- Avoid giving the opponent easy completions
- Maximize its score advantage

## Project structure

```
main.py             PPO training loop + Policy network
dots_and_boxes.py   PufferEnv + RaylibRenderer
pyproject.toml      Dependencies
```
