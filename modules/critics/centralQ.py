import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentralQCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralQCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if self.args.critic_nn == "deep":
            self.fc4 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.n_actions)       # returns Q-values
        return

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        last_hidden = F.relu(self.fc2(x))
        if self.args.critic_nn == "deep":
            last_hidden = F.relu(self.fc4(last_hidden))
        q = self.fc3(last_hidden)           # shape: (batch_size, maxseqlen, num_agents, num_actions)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size       # number of parallel envs
        max_t = batch.max_seq_length if t is None else 1        # maxseqlen
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state;    batch["state"]: (batch_size, maxseqlen, state_dim)
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))        # share state to all agents of the same env

        # observations;     batch["obs"]: (batch_size, maxseqlen, num_agents, obs_dim)      
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))      # share observations to all agents (i.e. state) of the same env

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        # Add agent_id in obs_dim (in one-hot representation)
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        # Concat inputs
        inputs = th.cat(inputs, dim=-1)     # shape: (batch_size, maxseqlen, num_agents, state_dim + num_agents ) ?? depends on flags
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents
        return input_shape