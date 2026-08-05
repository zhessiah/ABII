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

class NonSharedMAC:
    def __init__(self, scheme, groups, args, vae_controller=None):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None
        self.vae_controller = vae_controller

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t):

        if self.args.use_dynamics:

            predicted_states = th.zeros((self.args.batch_size, self.args.n_agents, self.vae_controller.state_dim)).to(self.args.device)       

            if not self.args.use_only_full_state_vae:
                obs = ep_batch["obs"][:, t, :, :]
                obs_mu = self.vae_controller.obs_ms.mean
                obs_std = th.sqrt(self.vae_controller.obs_ms.var) + 1e-8
                obs = (obs - obs_mu) / obs_std        # normalized observations   
                inputs = obs.view(self.args.batch_size, self.n_agents, -1)
                for agent_id in range(self.args.n_agents):
                    agent_predicted_states = self.vae_controller.forward(inputs[:, agent_id, :], agent_id, test_mode=True)
                    predicted_states[:, agent_id, :] = agent_predicted_states
            
            else:
                # use the state_vae for prediction instead of the agents' models
                inputs = ep_batch["state"][:, t, :].view(self.args.batch_size, -1)
                state_mu = self.vae_controller.state_ms.mean
                state_std = th.sqrt(self.vae_controller.state_ms.var) + 1e-8
                inputs = (inputs - state_mu) / state_std        # normalized states
                predicted_states = self.vae_controller.state_vae(inputs)[0]
                predicted_states = predicted_states.view(self.args.batch_size, 1, -1)
                predicted_states = predicted_states.repeat(1, self.args.n_agents, 1)

            # Unnormalize the predicted states
            state_mu = self.vae_controller.state_ms.mean
            state_std = th.sqrt(self.vae_controller.state_ms.var) + 1e-8
            predicted_states = predicted_states * state_std + state_mu

            if self.args.use_state_permutation:
                predicted_states = predicted_states.view(self.args.batch_size, 1, self.n_agents, -1)
                predicted_states = permutate_state(predicted_states, self.n_agents).view(self.args.batch_size, self.n_agents, -1)

            for agent_id in range(self.args.n_agents):
                obs_dim = self.vae_controller.obs_dim
                if self.args.use_state_permutation:
                    predicted_states[:, agent_id, 0:obs_dim] = ep_batch["obs"][:, t, agent_id, :]
                
                else:
                    predicted_states[:, agent_id, agent_id*obs_dim:agent_id*obs_dim+obs_dim] = ep_batch["obs"][:, t, agent_id, :]

            predicted_states = predicted_states.detach()
            agent_inputs = self._build_inputs(ep_batch, t, predicted_states)

        else:
            agent_inputs = self._build_inputs(ep_batch, t)

        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)       # pass hidden states from previous timestep

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)      # (batch_size, num_agents, num_actions)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav

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

    def _build_inputs(self, batch, t, predicted_states=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        # batch: EpisodeBatch. Batch Size:10 Max_seq_len:26 Keys:dict_keys(['state', 'obs', 'actions', 'avail_actions', 'reward', 'terminated', 'actions_onehot', 'filled']) Groups:dict_keys(['agents'])
        # batch["obs"].shape: torch.Size([batch_size, Max_seq_len, num_agents, obs_dim])
        # batch["obs"][:, t].shape: torch.Size([batch_size, num_agents, obs_dim])
        # batch["actions_onehot"].shape: torch.Size([batch_size, Max_seq_len, num_agents, num_actions])


        bs = batch.batch_size
        inputs = []
        
        if predicted_states is None:
            obs = batch["obs"][:, t]
            inputs.append(obs)  # b1av
        else:
            inputs.append(predicted_states)

        # Modify the observation shape
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)   
                
        return inputs       # Shape: (bs * n_agents) x obs_shape_modified


    def _get_input_shape(self, scheme):
        if self.args.use_dynamics:
            input_shape = scheme["state"]["vshape"]
        else:
            input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
