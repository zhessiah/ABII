import torch.nn as nn
from modules.agents.rnn_agent import RNNAgent
import torch as th

class RNNNSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNNSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.n_agents)])

    def init_hidden(self):
        # The hidden state is initialized per sub-agent, so we just need to call the first one.
        return self.agents[0].init_hidden()

    def forward(self, inputs, hidden_state):
        # Correctly split the inputs and hidden_state for each agent.
        # The input tensors have shape (bs * n_agents, -1).
        # We need to split them into n_agents chunks of size (bs, -1).
        
        bs = inputs.shape[0] // self.n_agents
        
        inputs_split = th.split(inputs, bs, dim=0)
        hidden_state_split = th.split(hidden_state, bs, dim=0)

        qs = []
        hs = []
        for i in range(self.n_agents):
            q, h = self.agents[i](inputs_split[i], hidden_state_split[i])
            qs.append(q)
            hs.append(h)

        # The output qs and hs are lists of tensors.
        # We need to concatenate them back into single tensors.
        q_outs = th.cat(qs, dim=0)
        h_outs = th.cat(hs, dim=0)
        return q_outs, h_outs

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
