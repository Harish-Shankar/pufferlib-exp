'''Dots and Boxes environment for PufferLib.

Two-player game on a grid of dots. Players take turns drawing lines between
adjacent dots. Completing the 4th side of a box scores a point and grants
another turn. The player with the most boxes wins.

The agent plays against a random opponent.
'''

import gymnasium
import numpy as np

import pufferlib


class DotsAndBoxes(pufferlib.PufferEnv):
    def __init__(self, rows=3, cols=3, render_mode='ansi', buf=None, seed=0):
        self.rows = rows
        self.cols = cols
        self.num_h_lines = (rows + 1) * cols      # horizontal lines
        self.num_v_lines = rows * (cols + 1)       # vertical lines
        self.num_lines = self.num_h_lines + self.num_v_lines
        self.num_boxes = rows * cols

        # Observation: lines_drawn | valid_mask | line_owner | box_owner
        obs_size = self.num_lines * 3 + self.num_boxes
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(obs_size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(self.num_lines)
        self.render_mode = render_mode
        self.num_agents = 1

        super().__init__(buf)

        # Internal state arrays
        self.lines = np.zeros(self.num_lines, dtype=np.uint8)       # 0/1 drawn
        self.line_owner = np.zeros(self.num_lines, dtype=np.uint8)  # 0=none,1=agent,2=opp
        self.box_owner = np.zeros(self.num_boxes, dtype=np.uint8)   # 0=none,1=agent,2=opp
        self.agent_score = 0
        self.opp_score = 0

        # Precompute which lines border each box
        self._box_lines = []
        for r in range(rows):
            for c in range(cols):
                top = r * cols + c
                bottom = (r + 1) * cols + c
                left = self.num_h_lines + r * (cols + 1) + c
                right = self.num_h_lines + r * (cols + 1) + c + 1
                self._box_lines.append((top, bottom, left, right))

    def _update_obs(self):
        nl = self.num_lines
        self.observations[0, :nl] = self.lines
        self.observations[0, nl:2*nl] = 1 - self.lines  # valid mask
        self.observations[0, 2*nl:3*nl] = self.line_owner
        self.observations[0, 3*nl:3*nl + self.num_boxes] = self.box_owner

    def reset(self, seed=0):
        self.lines[:] = 0
        self.line_owner[:] = 0
        self.box_owner[:] = 0
        self.agent_score = 0
        self.opp_score = 0
        self._update_obs()
        return self.observations, []

    def step(self, actions):
        action = actions[0]
        self.terminals[0] = False
        self.rewards[0] = 0.0

        # Invalid action: line already drawn
        if self.lines[action] == 1:
            self.rewards[0] = -0.1
            # Skip agent's turn, let opponent play
            self._opponent_turn()
            info = self._check_game_end()
            self._update_obs()
            return self.observations, self.rewards, self.terminals, self.truncations, info

        # Valid agent move
        self.lines[action] = 1
        self.line_owner[action] = 1
        boxes_completed = self._check_boxes(action, player=1)
        self.rewards[0] += boxes_completed * 1.0

        # Check if agent's move ended the game
        info = self._check_game_end()
        if info:
            self._update_obs()
            return self.observations, self.rewards, self.terminals, self.truncations, info

        # Extra turn if box completed
        if boxes_completed > 0:
            self._update_obs()
            return self.observations, self.rewards, self.terminals, self.truncations, []

        # Opponent's turn
        self._opponent_turn()
        info = self._check_game_end()
        self._update_obs()
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def _check_boxes(self, line_idx, player):
        completed = 0
        for i, (t, b, l, r) in enumerate(self._box_lines):
            if self.box_owner[i] != 0:
                continue
            if line_idx not in (t, b, l, r):
                continue
            if self.lines[t] and self.lines[b] and self.lines[l] and self.lines[r]:
                self.box_owner[i] = player
                if player == 1:
                    self.agent_score += 1
                else:
                    self.opp_score += 1
                completed += 1
        return completed

    def _opponent_turn(self):
        while True:
            valid = np.where(self.lines == 0)[0]
            if len(valid) == 0:
                return
            move = np.random.choice(valid)
            self.lines[move] = 1
            self.line_owner[move] = 2
            boxes = self._check_boxes(move, player=2)
            if boxes == 0 or self._game_over():
                return
            # Opponent gets extra turn on box completion

    def _game_over(self):
        return np.all(self.lines == 1)

    def _check_game_end(self):
        if not self._game_over():
            return []
        self.terminals[0] = True
        if self.agent_score > self.opp_score:
            self.rewards[0] += 3.0
        elif self.agent_score < self.opp_score:
            self.rewards[0] -= 3.0
        info = [{'reward': float(self.rewards[0]),
                 'agent_score': self.agent_score,
                 'opp_score': self.opp_score}]
        self.reset()
        return info

    def render(self):
        if self.render_mode == 'human':
            return None  # handled by RaylibRenderer externally
        rows, cols = self.rows, self.cols
        chars = []
        for r in range(rows + 1):
            # Draw horizontal lines
            line = ''
            for c in range(cols):
                line += '.'
                h_idx = r * cols + c
                if self.lines[h_idx]:
                    owner = self.line_owner[h_idx]
                    color = '94' if owner == 1 else '91'
                    line += f'\033[{color}m---\033[0m'
                else:
                    line += '   '
            line += '.'
            chars.append(line)

            # Draw vertical lines and box owners
            if r < rows:
                line = ''
                for c in range(cols + 1):
                    v_idx = self.num_h_lines + r * (cols + 1) + c
                    if self.lines[v_idx]:
                        owner = self.line_owner[v_idx]
                        color = '94' if owner == 1 else '91'
                        line += f'\033[{color}m|\033[0m'
                    else:
                        line += ' '
                    if c < cols:
                        box_idx = r * cols + c
                        if self.box_owner[box_idx] == 1:
                            line += ' \033[94mA\033[0m '
                        elif self.box_owner[box_idx] == 2:
                            line += ' \033[91mO\033[0m '
                        else:
                            line += '   '
                chars.append(line)

        chars.append(f'Agent: {self.agent_score}  Opponent: {self.opp_score}')
        return '\n'.join(chars)

    def close(self):
        pass


class RaylibRenderer:
    '''Raylib-based visual renderer for Dots and Boxes.'''

    def __init__(self, env, width=720, height=720, fps=3):
        from raylib import rl
        self.rl = rl
        self.env = env
        self.width = width
        self.height = height

        # Layout: margin around the grid, rest is playing field
        self.margin = 60
        self.header = 50  # space for score text at top

        rl.InitWindow(width, height, b"Dots and Boxes")
        rl.SetTargetFPS(fps)

    def _dot_pos(self, r, c):
        '''Get pixel position for dot at grid position (r, c).'''
        env = self.env
        field_w = self.width - 2 * self.margin
        field_h = self.height - 2 * self.margin - self.header
        x = self.margin + c * field_w // env.cols
        y = self.margin + self.header + r * field_h // env.rows
        return x, y

    def render(self):
        rl = self.rl
        env = self.env

        rl.BeginDrawing()
        rl.ClearBackground([30, 30, 40, 255])

        # Draw score header
        p1 = self.player1_name if hasattr(self, 'player1_name') else 'Agent'
        p2 = self.player2_name if hasattr(self, 'player2_name') else 'Opponent'
        score_text = f"{p1}: {env.agent_score}   {p2}: {env.opp_score}".encode()
        rl.DrawText(score_text, 20, 15, 24, [255, 255, 255, 255])

        # Draw filled boxes
        for idx, (t, b, l, r_line) in enumerate(env._box_lines):
            if env.box_owner[idx] == 0:
                continue
            row, col = divmod(idx, env.cols)
            x1, y1 = self._dot_pos(row, col)
            x2, y2 = self._dot_pos(row + 1, col + 1)
            if env.box_owner[idx] == 1:
                color = [70, 130, 230, 80]   # translucent blue
            else:
                color = [230, 70, 70, 80]    # translucent red
            rl.DrawRectangle(x1, y1, x2 - x1, y2 - y1, color)

        # Draw horizontal lines
        for r in range(env.rows + 1):
            for c in range(env.cols):
                h_idx = r * env.cols + c
                x1, y1 = self._dot_pos(r, c)
                x2, y2 = self._dot_pos(r, c + 1)
                if env.lines[h_idx]:
                    owner = env.line_owner[h_idx]
                    color = [100, 160, 255, 255] if owner == 1 else [255, 100, 100, 255]
                else:
                    color = [80, 80, 80, 255]
                thickness = 4.0 if env.lines[h_idx] else 2.0
                rl.DrawLineEx([x1, y1], [x2, y2], thickness, color)

        # Draw vertical lines
        for r in range(env.rows):
            for c in range(env.cols + 1):
                v_idx = env.num_h_lines + r * (env.cols + 1) + c
                x1, y1 = self._dot_pos(r, c)
                x2, y2 = self._dot_pos(r + 1, c)
                if env.lines[v_idx]:
                    owner = env.line_owner[v_idx]
                    color = [100, 160, 255, 255] if owner == 1 else [255, 100, 100, 255]
                else:
                    color = [80, 80, 80, 255]
                thickness = 4.0 if env.lines[v_idx] else 2.0
                rl.DrawLineEx([x1, y1], [x2, y2], thickness, color)

        # Draw dots
        for r in range(env.rows + 1):
            for c in range(env.cols + 1):
                x, y = self._dot_pos(r, c)
                rl.DrawCircleV([x, y], 6.0, [255, 255, 255, 255])

        # Turn indicator (for play mode)
        if hasattr(self, 'turn_text') and self.turn_text and not env.terminals[0]:
            text_w = rl.MeasureText(self.turn_text.encode(), 20)
            rl.DrawText(self.turn_text.encode(),
                        self.width - text_w - 20, 18, 20, [200, 200, 200, 255])

        # Game over flash
        if env.terminals[0]:
            if env.agent_score > env.opp_score:
                msg = f"{p1} Wins!".encode()
            elif env.agent_score < env.opp_score:
                msg = f"{p2} Wins!".encode()
            else:
                msg = b"Draw!"
            rl.DrawText(msg, self.width // 2 - 80, self.height // 2, 30, [255, 255, 0, 255])

        rl.EndDrawing()

    def line_from_click(self, mx, my):
        '''Return the index of the undrawn line closest to (mx, my), or -1.'''
        env = self.env
        best_idx = -1
        best_dist = 20.0  # max click distance in pixels

        # Check horizontal lines
        for r in range(env.rows + 1):
            for c in range(env.cols):
                h_idx = r * env.cols + c
                if env.lines[h_idx]:
                    continue
                x1, y1 = self._dot_pos(r, c)
                x2, y2 = self._dot_pos(r, c + 1)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_idx = h_idx

        # Check vertical lines
        for r in range(env.rows):
            for c in range(env.cols + 1):
                v_idx = env.num_h_lines + r * (env.cols + 1) + c
                if env.lines[v_idx]:
                    continue
                x1, y1 = self._dot_pos(r, c)
                x2, y2 = self._dot_pos(r + 1, c)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_idx = v_idx

        return best_idx

    def should_close(self):
        return self.rl.WindowShouldClose()

    def close(self):
        self.rl.CloseWindow()


def _load_policy(checkpoint, env):
    '''Load a trained policy from a checkpoint file.'''
    import torch
    from main import Policy
    obs_size = env.single_observation_space.shape[0]
    act_size = env.single_action_space.n
    policy = Policy(obs_size, act_size)
    policy.load_state_dict(torch.load(checkpoint, weights_only=True))
    policy.eval()
    print(f'Loaded checkpoint: {checkpoint}')
    return policy


def _ai_pick_action(policy, env):
    '''Use the trained policy to pick a line index.'''
    import torch
    env._update_obs()
    obs_t = torch.tensor(env.observations, dtype=torch.float32)
    with torch.no_grad():
        logits, _ = policy(obs_t)
        # Mask invalid actions (already drawn lines)
        mask = torch.tensor(env.lines, dtype=torch.bool)
        logits[0, mask] = float('-inf')
        action = torch.distributions.Categorical(logits=logits).sample().item()
    return action


def play_mode(args):
    '''Human vs AI interactive play mode.'''
    import time

    if args.checkpoint is None:
        print('Error: --checkpoint is required for --play mode')
        return

    env = DotsAndBoxes(rows=args.rows, cols=args.cols, render_mode='human')
    env.reset()
    renderer = RaylibRenderer(env, fps=60)
    renderer.player1_name = 'You'
    renderer.player2_name = 'AI'
    policy = _load_policy(args.checkpoint, env)
    rl = renderer.rl

    human_turn = True
    game_over = False
    game_over_time = 0

    print('Click on a line to draw it. You are Blue, AI is Red.')

    while not renderer.should_close():
        # After game over, pause then reset
        if game_over:
            renderer.render()
            if time.time() - game_over_time > 2.0:
                env.reset()
                human_turn = True
                game_over = False
            continue

        renderer.turn_text = 'Your turn' if human_turn else 'AI thinking...'

        if human_turn:
            # Wait for mouse click
            if rl.IsMouseButtonPressed(0):
                mx, my = rl.GetMouseX(), rl.GetMouseY()
                line_idx = renderer.line_from_click(mx, my)
                if line_idx >= 0:
                    env.lines[line_idx] = 1
                    env.line_owner[line_idx] = 1
                    boxes = env._check_boxes(line_idx, player=1)

                    if env._game_over():
                        env.terminals[0] = True
                        game_over = True
                        game_over_time = time.time()
                    elif boxes == 0:
                        human_turn = False
                    # else: extra turn for human
                    env._update_obs()
        else:
            # AI turn loop: AI keeps going while it completes boxes
            action = _ai_pick_action(policy, env)
            env.lines[action] = 1
            env.line_owner[action] = 2
            boxes = env._check_boxes(action, player=2)
            env._update_obs()
            renderer.render()

            if env._game_over():
                env.terminals[0] = True
                game_over = True
                game_over_time = time.time()
            elif boxes == 0:
                human_turn = True
            # else: AI gets another turn (will loop next frame)

        renderer.render()

    renderer.close()


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true', help='Run SPS benchmark')
    parser.add_argument('--play', action='store_true',
                        help='Play against the trained AI (requires --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained policy checkpoint (.pt)')
    parser.add_argument('--fps', type=int, default=3, help='Render FPS')
    parser.add_argument('--rows', type=int, default=3)
    parser.add_argument('--cols', type=int, default=3)
    args = parser.parse_args()

    if args.benchmark:
        env = DotsAndBoxes(rows=args.rows, cols=args.cols)
        env.reset()
        steps = 0
        CACHE = 1024
        actions = np.random.randint(0, env.num_lines, (CACHE, 1))
        start = time.time()
        while time.time() - start < 10:
            obs, rewards, terminals, truncations, info = env.step(actions[steps % CACHE])
            steps += 1
        print(f'DotsAndBoxes SPS: {int(steps / (time.time() - start))}')
    elif args.play:
        play_mode(args)
    else:
        env = DotsAndBoxes(rows=args.rows, cols=args.cols, render_mode='human')
        env.reset()
        renderer = RaylibRenderer(env, fps=args.fps)

        # Load trained policy if checkpoint provided
        policy = None
        if args.checkpoint:
            policy = _load_policy(args.checkpoint, env)

        while not renderer.should_close():
            if policy is not None:
                import torch
                obs_t = torch.tensor(env.observations, dtype=torch.float32)
                with torch.no_grad():
                    logits, _ = policy(obs_t)
                    action = torch.distributions.Categorical(logits=logits).sample().numpy()
            else:
                action = np.random.randint(0, env.num_lines, (1,))
            env.step(action)
            renderer.render()

        renderer.close()
