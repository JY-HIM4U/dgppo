import jax.numpy as jnp
import jax.tree_util as jtu
import jax
import numpy as np
import socket
import matplotlib.pyplot as plt
import os

from typing import Callable, TYPE_CHECKING, Union
from matplotlib.colors import CenteredNorm

from ..utils.typing import PRNGKey, Array
from .data import Rollout


if TYPE_CHECKING:
    from ..env import MultiAgentEnv
else:
    MultiAgentEnv = None


def rollout(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
) -> Union[Rollout, list[Rollout]]:
    """
    Get a rollout from the environment using the actor.

    Parameters
    ----------
    env: MultiAgentEnv
    actor: Callable, [GraphsTuple, Array, RNN_States, PRNGKey] -> [Action, LogPi, RNN_States]
    init_rnn_state: Array
    key: PRNGKey

    Returns
    -------
    data: Rollout or list[Rollout] - If multiple groups, returns list of Rollouts (one per group)
    """
    key_x0, key_z0, key = jax.random.split(key, 3)
    init_graphs = env.reset(key_x0)
    
    # Handle both single graph and list of graphs (multiple groups)
    if isinstance(init_graphs, list):
        # Multiple groups: process all groups together so they interact
        num_groups = len(init_graphs)
        
        def body(data, key_):
            # data contains: (list of graphs, list of rnn_states)
            graphs, rnn_states = data
            
            # Get actions for all groups
            actions_list = []
            log_pis_list = []
            new_rnn_states_list = []
            for graph, rnn_state in zip(graphs, rnn_states):
                action, log_pi, new_rnn_state = actor(graph, rnn_state, key_)
                actions_list.append(action)
                log_pis_list.append(log_pi)
                new_rnn_states_list.append(new_rnn_state)
            
            # Step all groups together (they interact through shared obstacles and lidar)
            # Pass all graphs and actions as lists to env.step
            next_graphs_list, rewards_list, costs_list, dones_list, infos_list = env.step(graphs, actions_list)
            
            return ((next_graphs_list, new_rnn_states_list),
                    (graphs, actions_list, rnn_states, rewards_list, costs_list, dones_list, log_pis_list, next_graphs_list))
        
        keys = jax.random.split(key, env.max_episode_steps)
        init_rnn_states = [init_rnn_state] * num_groups
        _, (graphs_list, actions_list, rnn_states_list, rewards_list, costs_list, dones_list, log_pis_list, next_graphs_list) = (
            jax.lax.scan(body, (init_graphs, init_rnn_states), keys, length=env.max_episode_steps))
        
        # Convert from time-major lists to group-major rollouts
        # graphs_list: (T, num_groups) -> we want (num_groups, T)
        rollout_datas = []
        for group_idx in range(num_groups):
            graphs = [graphs_list[t][group_idx] for t in range(env.max_episode_steps)]
            actions = [actions_list[t][group_idx] for t in range(env.max_episode_steps)]
            rnn_states = [rnn_states_list[t][group_idx] for t in range(env.max_episode_steps)]
            rewards = [rewards_list[t][group_idx] for t in range(env.max_episode_steps)]
            costs = [costs_list[t][group_idx] for t in range(env.max_episode_steps)]
            dones = [dones_list[t][group_idx] for t in range(env.max_episode_steps)]
            log_pis = [log_pis_list[t][group_idx] for t in range(env.max_episode_steps)]
            next_graphs = [next_graphs_list[t][group_idx] for t in range(env.max_episode_steps)]
            
            # Stack into arrays
            graphs = jnp.stack(graphs)
            actions = jnp.stack(actions)
            rnn_states = jnp.stack(rnn_states)
            rewards = jnp.stack(rewards)
            costs = jnp.stack(costs)
            dones = jnp.stack(dones)
            log_pis = jnp.stack(log_pis) if log_pis[0] is not None else None
            next_graphs = jnp.stack(next_graphs)
            
            rollout_data = Rollout(graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs)
            rollout_datas.append(rollout_data)
        return rollout_datas
    else:
        # Single graph: original behavior
        init_graph = init_graphs
        
        def body(data, key_):
            graph, rnn_state = data
            action, log_pi, new_rnn_state = actor(graph, rnn_state, key_)
            next_graph, reward, cost, done, info = env.step(graph, action)
            return ((next_graph, new_rnn_state),
                    (graph, action, rnn_state, reward, cost, done, log_pi, next_graph))
        
        keys = jax.random.split(key, env.max_episode_steps)
        _, (graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs) = (
            jax.lax.scan(body, (init_graph, init_rnn_state), keys, length=env.max_episode_steps))
        rollout_data = Rollout(graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs)
        return rollout_data


def test_rollout(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        stochastic: bool = False
) -> Union[Rollout, list[Rollout]]:
    """
    Get a test rollout from the environment using the actor.
    
    Parameters
    ----------
    env: MultiAgentEnv
    actor: Callable, [GraphsTuple, Array, RNN_States, PRNGKey] -> [Action, RNN_States]
    init_rnn_state: Array
    key: PRNGKey
    stochastic: bool, whether to use stochastic actions
    
    Returns
    -------
    data: Rollout or list[Rollout] - If multiple groups, returns list of Rollouts (one per group)
    """
    key_x0, key = jax.random.split(key)
    init_graphs = env.reset(key_x0)
    
    # Handle both single graph and list of graphs (multiple groups)
    if isinstance(init_graphs, list):
        # Multiple groups: process all groups together so they interact
        num_groups = len(init_graphs)
        
        def body_(data, key_):
            # data contains: (list of graphs, list of rnn_states)
            graphs, rnn_states = data
            
            # Get actions for all groups
            actions_list = []
            new_rnn_states_list = []
            for graph, rnn_state in zip(graphs, rnn_states):
                if not stochastic:
                    action, new_rnn_state = actor(graph, rnn_state)
                else:
                    action, new_rnn_state = actor(graph, rnn_state, key_)
                actions_list.append(action)
                new_rnn_states_list.append(new_rnn_state)
            
            # Step all groups together (they interact through shared obstacles and lidar)
            # Pass all graphs and actions as lists to env.step
            next_graphs_list, rewards_list, costs_list, dones_list, infos_list = env.step(graphs, actions_list)
            
            # Return same structure as rollout (with log_pis_list as None for test_rollout)
            log_pis_list = [None] * num_groups  # Dummy log_pis for test_rollout
            return ((next_graphs_list, new_rnn_states_list),
                    (graphs, actions_list, rnn_states, rewards_list, costs_list, dones_list, log_pis_list, next_graphs_list))
        
        keys = jax.random.split(key, env.max_episode_steps)
        init_rnn_states = [init_rnn_state] * num_groups
        
        # DEBUG: Print initial structure
        print(f"[DEBUG test_rollout] num_groups: {num_groups}")
        print(f"[DEBUG test_rollout] len(init_graphs): {len(init_graphs)}")
        print(f"[DEBUG test_rollout] type(init_graphs): {type(init_graphs)}")
        print(f"[DEBUG test_rollout] type(init_graphs[0]): {type(init_graphs[0]) if len(init_graphs) > 0 else 'N/A'}")
        print(f"[DEBUG test_rollout] len(init_rnn_states): {len(init_rnn_states)}")
        print(f"[DEBUG test_rollout] env.max_episode_steps: {env.max_episode_steps}")
        
        _, (graphs_list, actions_list, rnn_states_list, rewards_list, costs_list, dones_list, log_pis_list, next_graphs_list) = (
            jax.lax.scan(body_,
                        (init_graphs, init_rnn_states),
                        keys,
                        length=env.max_episode_steps))
        
        # DEBUG: Print scan output structure
        print(f"[DEBUG test_rollout] After scan:")
        print(f"  type(graphs_list): {type(graphs_list)}")
        print(f"  len(graphs_list): {len(graphs_list) if hasattr(graphs_list, '__len__') else 'N/A'}")
        if hasattr(graphs_list, '__len__') and len(graphs_list) > 0:
            print(f"  type(graphs_list[0]): {type(graphs_list[0])}")
            if hasattr(graphs_list[0], '__len__'):
                print(f"  len(graphs_list[0]): {len(graphs_list[0])}")
            if hasattr(graphs_list[0], 'nodes'):
                print(f"  graphs_list[0].nodes.shape: {graphs_list[0].nodes.shape if hasattr(graphs_list[0].nodes, 'shape') else 'N/A'}")
        
        print(f"  type(actions_list): {type(actions_list)}")
        print(f"  len(actions_list): {len(actions_list) if hasattr(actions_list, '__len__') else 'N/A'}")
        if hasattr(actions_list, '__len__') and len(actions_list) > 0:
            print(f"  type(actions_list[0]): {type(actions_list[0])}")
            if hasattr(actions_list[0], '__len__'):
                print(f"  len(actions_list[0]): {len(actions_list[0])}")
            if hasattr(actions_list[0], 'shape'):
                print(f"  actions_list[0].shape: {actions_list[0].shape}")
        
        print(f"  type(rewards_list): {type(rewards_list)}")
        print(f"  len(rewards_list): {len(rewards_list) if hasattr(rewards_list, '__len__') else 'N/A'}")
        if hasattr(rewards_list, '__len__') and len(rewards_list) > 0:
            print(f"  type(rewards_list[0]): {type(rewards_list[0])}")
            if hasattr(rewards_list[0], '__len__'):
                print(f"  len(rewards_list[0]): {len(rewards_list[0])}")
            if hasattr(rewards_list[0], 'shape'):
                print(f"  rewards_list[0].shape: {rewards_list[0].shape}")
        
        # Convert from group-major stacked structure to group-major rollouts
        # jax.lax.scan stacks the outputs, so:
        # graphs_list: list of length num_groups, where each element is a stacked GraphsTuple (T, ...)
        # actions_list: list of length num_groups, where each element is a stacked array (T, ...)
        # The stacked GraphsTuple is already in the correct format for Rollout
        rollout_datas = []
        for group_idx in range(num_groups):
            # graphs_list[group_idx] is already a stacked GraphsTuple with shape (T, ...)
            # This is exactly what Rollout expects - no need to unstack and re-stack
            graphs = graphs_list[group_idx]  # Already stacked GraphsTuple (T, ...)
            
            # For arrays, they're already stacked: (T, ...)
            # Just extract the group_idx element (which is already the right group)
            actions = actions_list[group_idx]  # Shape: (T, num_agents, action_dim)
            rnn_states = rnn_states_list[group_idx]  # Shape: (T, ...)
            rewards = rewards_list[group_idx]  # Shape: (T,)
            costs = costs_list[group_idx]  # Shape: (T, num_costs) or (T,)
            dones = dones_list[group_idx]  # Shape: (T,)
            
            # next_graphs_list[group_idx] is also already a stacked GraphsTuple
            next_graphs = next_graphs_list[group_idx]  # Already stacked GraphsTuple (T, ...)
            
            rollout_data = Rollout(graphs, actions, rnn_states, rewards, costs, dones, None, next_graphs)
            rollout_datas.append(rollout_data)
        return rollout_datas
    else:
        # Single graph: original behavior
        init_graph = init_graphs
        
        def body_(data, key_):
            graph, rnn_state = data
            if not stochastic:
                action, rnn_state = actor(graph, rnn_state)
            else:
                action, rnn_state = actor(graph, rnn_state, key_)
            next_graph, reward, cost, done, info = env.step(graph, action)
            return (next_graph, rnn_state), (graph, action, rnn_state, reward, cost, done, None, next_graph)

        keys = jax.random.split(key, env.max_episode_steps)
        _, (graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs) = (
            jax.lax.scan(body_,
                        (init_graph, init_rnn_state),
                        keys,
                        length=env.max_episode_steps))
        rollout_data = Rollout(graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs)
        return rollout_data


def has_nan(x):
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def has_inf(x):
    return jtu.tree_map(lambda y: jnp.isinf(y).any(), x)


def has_any_inf(x):
    return jnp.array(jtu.tree_flatten(has_inf(x))[0]).any()


def has_any_nan_or_inf(x):
    return has_any_nan(x) | has_any_inf(x)


def compute_norm(grad):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def tree_copy(tree):
    return jtu.tree_map(lambda x: x.copy(), tree)


def jax2np(x):
    return jtu.tree_map(lambda y: np.array(y), x)


def np2jax(x):
    return jtu.tree_map(lambda y: jnp.array(y), x)


def internet(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(ex)
        return False


def is_connected():
    return internet()


def centered_norm(vmin: float | list[float], vmax: float | list[float]):
    if isinstance(vmin, list):
        vmin = min(vmin)
    if isinstance(vmax, list):
        vmin = max(vmax)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(0, halfrange)


def plot_rnn_states(rnn_states: Array, name: str, path: str):
    """
    rnn_states: (T, n_layer, n_agent, n_carry, hid_size)
    """
    T, n_layer, n_agent, n_carry, hid_size = rnn_states.shape
    for i_layer in range(n_layer):
        fig, ax = plt.subplots(nrows=n_agent, ncols=n_carry, figsize=(10, 20))
        for i_agent in range(n_agent):
            for i_carry in range(n_carry):
                ax[i_agent, i_carry].plot(rnn_states[:, i_layer, i_agent, i_carry, :])
                ax[i_agent, i_carry].set_title(f'Agent {i_agent}, carry {i_carry}, layer {i_layer}')
                ax[i_agent, i_carry].set_xlabel('Time step')
                ax[i_agent, i_carry].set_ylabel('State value')
        fig.tight_layout()
        plt.savefig(os.path.join(path, f'rnn_states_{name}_layer{i_layer}.png'))
