import torch
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL

class MultiHumanRL(CADRL):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        ***MODIFIED***: We remove the usage of 'self.env.onestep_lookahead(...)' 
        to avoid 'NoneType' errors. Instead, we do a "fake" next human states 
        via self.propagate() + compute_reward(...) for each candidate action.
        """

        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        # if robot reached goal
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)

        # build action space if not exist
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()

        # e-greedy
        if self.phase == 'train' and probability < self.epsilon:
            # random
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            # For each action: propagate robot, also propagate humans 
            # with their current velocities, compute reward, then feed into net 
            self.action_values = []
            max_value = float('-inf')
            max_action = None

            for action in self.action_space:
                # 1) propagate robot
                next_self_state = self.propagate(state.self_state, action)
                # 2) propagate each human
                next_human_states = [
                    self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                    for human_state in state.human_states
                ]
                # 3) compute reward
                reward = self.compute_reward(next_self_state, next_human_states)

                # 4) build batch input to network 
                batch_next_states = torch.cat([
                    torch.Tensor([next_self_state + next_human_state]).to(self.device)
                    for next_human_state in next_human_states
                ], dim=0)

                # 5) rotate / occupancy
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)

                # 6) forward net => value
                next_state_value = self.model(rotated_batch_input).data.item()

                # 7) combine immediate reward + discount factor
                #    note: we use (gamma^( time_step * v_pref ))
                #    you can tweak if you want
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value

                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action

            if max_action is None:
                raise ValueError('Value network is not well trained or action_values are all -inf?')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def compute_reward(self, nav, humans):
        """
        Evaluate simple collision or goal-based reward.
        'nav' is next robot state, 'humans' are next human states.
        """
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius

        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1.0
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0.0

        return reward

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network
        (# of humans, len(state)).
        """
        state_tensor = torch.cat([
            torch.Tensor([state.self_state + human_state]).to(self.device)
            for human_state in state.human_states
        ], dim=0)

        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps.to(self.device)], dim=1)
        else:
            state_tensor = self.rotate(state_tensor)
        return state_tensor

    def input_dim(self):
        return self.joint_state_dim + (
            (self.cell_num ** 2) * self.om_channel_size if self.with_om else 0
        )

    def build_occupancy_maps(self, human_states):
        """
        occupancy map usage if with_om = True
        """
        occupancy_maps = []
        for human in human_states:
            # remove 'other_humans' if needed or keep as is
            other_humans = np.concatenate([
                np.array([(oh.px, oh.py, oh.vx, oh.vy)])
                for oh in human_states if oh != human
            ], axis=0)

            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)

            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # compute velocity channel
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed

                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for idx_cell, cell_list in enumerate(dm):
                    dm[idx_cell] = sum(cell_list) / len(cell_list) if len(cell_list) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

