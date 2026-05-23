from argparse import ArgumentTypeError

from torch.utils.tensorboard import SummaryWriter

from .imports import *

seed = 0  # Global variable to store the seed


# Taking np.array from gym and casting into tensor
def cast_np_to_tensors(dict_np, device="cpu", dtype=th.float32):
    return {
        agent_id: th.tensor(obs, dtype=dtype, device=device)
        for agent_id, obs in dict_np.items()
    }


# Creating the joint obs from the tensor dict
def stack_agent_obs_by_env(obs_dict):
    # Just concatenate all agent tensors on the last dimension
    return th.cat(list(obs_dict.values()), dim=-1)


# Creating a list of dicts for each env's actions
def split_action_tensor_dict(action_dict):
    return [dict(zip(action_dict, t)) for t in zip(*action_dict.values())]


def set_torch(
    n_threads: int = 0, deterministic: bool = True, cuda: bool = False
) -> th.device:
    """Configure PyTorch settings including the number of threads, determinism, and CUDA usage.

    Args:
        n_threads: Number of threads for PyTorch operations.
        deterministic: Whether to use deterministic algorithms.
        cuda: Whether to enable CUDA.

    Returns:
        The PyTorch device configured based on availability and input flags.
    """
    th.set_num_threads(n_threads)
    th.backends.cudnn.deterministic = deterministic
    if cuda and th.cuda.is_available():
        return th.device("cuda")
    if cuda and th.backends.mps.is_available():
        return th.device("mps")
    return th.device("cpu")


def set_random_seed(s: Optional[int] = None) -> None:
    """Set the random seed for reproducibility across different libraries.

    Args:
        s: Seed value to set. If None, the global seed value is used.
    """
    global seed
    if s is not None:
        seed = s
    rnd.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def str2bool(s: str) -> bool:
    """Convert a string representation of a boolean to an actual boolean value.

    Args:
        s: String to convert.

    Returns:
        The boolean value corresponding to the input string.

    Raises:
        ArgumentTypeError: If the string does not represent a boolean value.
    """
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    raise ArgumentTypeError("Boolean value expected.")


def Linear(
    input_dim: int,
    output_dim: int,
    act_fn: str = "relu",
    init_weight_uniform: bool = True,
) -> nn.Linear:
    """Create and initialize a linear layer with appropriate weights.
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        act_fn: Activation function.
        init_weight_uniform: Whether to uniformly sample initial weights.

    Returns:
        The initialized layer.
    """
    act_fn = act_fn.lower()
    gain = nn.init.calculate_gain(act_fn)
    layer = nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(layer.weight, gain=gain)
    else:
        nn.init.xavier_normal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.00)
    return layer


class Tanh(nn.Module):
    """In-place tanh module."""

    def forward(self, input: th.Tensor) -> th.Tensor:
        """Forward pass for the in-place tanh activation function.

        Args:
            input (th.Tensor): Input tensor.

        Returns:
            Output tensor after applying in-place tanh.
        """
        return th.tanh_(input)


# Useful for picking up the right activation in the networks
th_act_fns = {
    "tanh": Tanh(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
}


class RunningMeanStd:
    """Welford-style running estimator of scalar mean/variance, vectorized over batches.

    Used to track the running variance of discounted returns for reward normalization
    (Stable Baselines3 VecNormalize style).
    """

    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(eps)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        batch_count = x.size
        if batch_count == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / new_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / new_count
        self.mean = new_mean
        self.var = M2 / new_count
        self.count = new_count


class ReturnNormalizer:
    """Reward normalization by running std of discounted returns.

    Per-env, this tracks G_t = gamma * G_{t-1} + r_t, then maintains a running
    estimate of Var[G]. At each step, the raw reward r_t is divided by sqrt(Var[G])
    before being stored in the rollout buffer. Mean is NOT subtracted (would shift
    the optimal policy). On episode end, the per-env return tracker is reset.

    Reference: Stable Baselines3 VecNormalize (https://stable-baselines3.readthedocs.io).
    """

    def __init__(self, n_envs: int, gamma: float, eps: float = 1e-8):
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.ret_rms = RunningMeanStd()
        self.returns = np.zeros(int(n_envs), dtype=np.float64)

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        reward = np.asarray(reward, dtype=np.float64)
        done = np.asarray(done).astype(bool)
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns)
        normed = reward / np.sqrt(self.ret_rms.var + self.eps)
        self.returns[done] = 0.0
        return normed.astype(np.float32)
