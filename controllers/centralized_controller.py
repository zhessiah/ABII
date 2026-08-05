from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


def permutate_state(state_t, n_agents):

    last_dim = state_t.shape[3]
    for agent_id in range(n_agents):
        cut = int(last_dim/n_agents)
        temp = state_t[:, :, agent_id, 0:cut].clone()
        state_t[:, :, agent_id, 0:cut] = state_t[:, :, agent_id, agent_id*cut:agent_id*cut+cut]
        state_t[:, :, agent_id, agent_id*cut:agent_id*cut+cut] = temp

    return state_t


# This multi-agent controller shares parameters between agents
class CentralizedMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # ep_batch["avail_actions"].shape: torch.Size([batch_size, Maxseqlen, num_agents, num_actions])
        # Only select actions for the selected batch elements in bs        
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)       # pass hidden states from previous timesteps

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)      # (batch_size, num_agents, num_actions)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)    # Shape: batch_size x self.n_agents x -1

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size       # number of parallel envs
        max_t = batch.max_seq_length if t is None else 1        # maxseqlen
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state;    batch["state"]: (batch_size, maxseqlen, state_dim)
        state_t = batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        
        if self.args.use_state_permutation: state_t = permutate_state(state_t, self.n_agents)
        
        inputs.append(state_t)       

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
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=-1)     # shape: (batch_size, maxseqlen, num_agents, state_dim + num_agents ) ?? depends on flags
        return inputs       # Shape: (bs * n_agents) x obs_shape_modified

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
