from .rnn_agent import RNNAgent

class AdvRNNAgent(RNNAgent):
    """
    Adversarial agent's policy network.
    It inherits from RNNAgent and has an identical architecture,
    but it will be trained with a different objective, resulting in different weights.
    Having a separate class allows for future specialization of the adversarial agent's behavior.
    """
    def __init__(self, input_shape, args):
        super(AdvRNNAgent, self).__init__(input_shape, args)
