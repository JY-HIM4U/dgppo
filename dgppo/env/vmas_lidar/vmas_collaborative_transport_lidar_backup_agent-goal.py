import pathlib
import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax.random as jr
import functools as ft
from jaxtyping import Float

from typing import NamedTuple, Optional, Tuple, Dict, List
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from .physax.entity import Agent, Entity
from .physax.shapes import Box, Sphere, Object
from .physax.world import World
from dgppo.trainer.data import Rollout
from matplotlib.patches import Polygon
from dgppo.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from dgppo.utils.typing import Action, Array, Cost, Done, Info, Reward, State, PRNGKey, Pos2d
from dgppo.utils.utils import save_anim, tree_index, jax_vmap, merge01
from dgppo.env.base import MultiAgentEnv
from dgppo.env.utils import get_node_goal_rng, get_lidar
from dgppo.env.obstacle import Rectangle, RECTANGLE, Obstacle
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
jax.config.update('jax_platform_name', 'cpu')


class VMASCollaborativeTransportLidarState(NamedTuple):
    agent: State       # Agent states (positions and velocities)
    object: State      # Object state (position, velocity, angle, angular velocity)
    goal: State        # Goal position
    obstacle: Obstacle # Obstacles in the environment
    
    @property
    def a_pos(self):
        return self.agent[:, :2]
    
    @property
    def a_vel(self):
        return self.agent[:, 2:4]
    
    @property
    def object_pos(self):
        object = self.object if self.object.ndim == 2 else self.object.reshape((1, -1))
        return object[:, :2]
    
    @property
    def object_vel(self):
        return self.object[:, 2:4]
    
    @property
    def object_angle(self):
        return self.object[:, 4:5]
    
    @property
    def object_angvel(self):
        return self.object[:, 5:6]
    
    @property
    def goal_pos(self):
        goal = self.goal if self.goal.ndim == 2 else self.goal.reshape((1, -1))
        return goal[:, :2]

LidarEnvGraphsTuple = GraphsTuple[State, VMASCollaborativeTransportLidarState]

class VMASCollaborativeTransportLidar(MultiAgentEnv):
    AGENT = 0
    GOAL = 1
    OBS = 2
    OBJECT = 3

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "lidar_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.2, 0.5],
        "top_k_rays": 8,
        "n_obs": 3,
        "default_area_size": 1.0
    }

    def __init__(
            self,
            num_agents: int = 3,
            num_obstacles: int = None,
            num_objects: int = 1,
            n_obs: int = None,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None,
            local_only: bool = False,
            object_length: float = 0.1,
            object_mass: float = 10.0,
            half_width: float = 0.8
    ):
        area_size_value = area_size if area_size is not None else self.PARAMS['default_area_size']
        super().__init__(num_agents, area_size=area_size_value, dt=dt)
        # Use a different attribute name (e.g. _params) to store the configuration.
        self._params = dict(self.PARAMS)
        if params is not None:
            self._params.update(params)
        self.object_length = object_length
        self.object_mass = object_mass
        self.half_width = half_width
        self.num_goals = self.num_agents
        self.num_objects = num_objects
        self.num_obstacles = num_obstacles
        self.n_obs = self._params["n_obs"]
        self.max_step = max_step
        self.local_only = local_only
        self.step_count = 0
        
        self.agent_radius = self._params["car_radius"]
        self.comm_radius = self._params["comm_radius"]
        self.lidar_radius = self._params["lidar_radius"]
        self.n_rays = self._params["n_rays"]
        self.n_rays = self._params.get("n_rays", 32)

        self.obs_len_range = self._params["obs_len_range"]
        self.top_k_rays = self._params["top_k_rays"]
        self.create_obstacles = jax_vmap(Rectangle.create)
        
        self.object_radius = 0.15
        self.object_mass = 1.0
        self.agent_mass = 0.5
        self.object_friction = 0.2
        
        self._node_dim = 10  # [x, y, vx, vy] # state dim (6) + indicator: agent: 0001, goal: 0010, object: 0100, obstacle: 1000
        self._edge_dim = 6  # [rel_x, rel_y, rel_vx, rel_vy]
        
        self._action_dim = 2  # [force_x, force_y]
        
        self.reward_scale = 1.0
        self.goal_threshold = 0.1

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def node_dim(self) -> int:
        return self._node_dim

    @property
    def edge_dim(self) -> int:
        return self._edge_dim

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def object_dim(self) -> int:
        return 6

    @property
    def n_cost(self) -> int:
        return 3

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obstacle collisions"

    def reset(self, key: Array)  -> GraphsTuple:
        """Reset the environment."""
        object_key, goal_key, obstacle_key,obstacle_theta_key  = jax.random.split(key, 4)
        n_rng_obs= self.n_obs
        if n_rng_obs == 0:
            obstacles = None
        else:
            obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
            length_key, key = jr.split(key, 2)
            obs_len = jr.uniform(
                length_key,
                (n_rng_obs, 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            obs_theta = jr.uniform(obstacle_theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)
            
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, self.num_objects, 2.2 * self.object_length, obstacles)
        obj_pos= states
        
        obj_angle = jax.random.uniform(object_key, minval=0.0, maxval=2 * np.pi)
        angles = jnp.array([obj_angle + i * 2 * jnp.pi / 3 for i in range(3)])
        agent_pos = obj_pos + self.object_length * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        obj_cen_halfwidth = self.half_width - self.object_length
        obj_radius = 0.98 * obj_cen_halfwidth
        # Initialize velocities as zero
        obj_vel = jnp.zeros(2)
        obj_angvel = jnp.array(0.0)
        agent_vel = jnp.zeros((self.num_agents, 2))
        agent_state = jnp.zeros((self.num_agents, self.state_dim), dtype=jnp.float32)
        agent_state = agent_state.at[:, :2].set(agent_pos)
        agent_state = agent_state.at[:, 2:4].set(agent_vel)
        
        # Sample goal position opposite to object
        goal_center = goals

        # Set the number of goals equal to the number of agents.
        self.num_goals = self.num_agents

        # Compute three vertices for the goal.
        # These vertices form an equilateral triangle centered at goal_center.
        # The distance from goal_center to each vertex is self.object_length.
        angles = jnp.array([0, 2 * jnp.pi / 3, 4 * jnp.pi / 3])  # for three vertices
        goal_vertices = goal_center + self.object_length * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

        # Goal state with static dimensions (state_dim, e.g., 6).
        goal_state = jnp.zeros((self.num_goals, self.state_dim), dtype=jnp.float32)
        goal_state = goal_state.at[:, :2].set(goal_vertices)
        
         # Object state with static dimensions
        object_state = jnp.zeros((self.num_objects, self.state_dim), dtype=jnp.float32)
        object_state = object_state.at[:,:2].set(obj_pos)
        object_state = object_state.at[:,2:4].set(obj_vel)
        object_state = object_state.at[:,4:5].set(obj_angle)
        object_state = object_state.at[:,5:6].set(obj_angvel)
        
        assert agent_state.shape == (self.num_agents, self.state_dim)
        assert goal_state.shape == (self.num_goals, self.state_dim)
        assert object_state.shape == (self.num_objects, self.state_dim)
        
        # Create the state object
        init_state = VMASCollaborativeTransportLidarState(
            agent=agent_state,
            goal=goal_state,
            object=object_state,
            obstacle=obstacles
        )
        
        lidar_data = self.get_lidar_data(init_state.agent, init_state.obstacle)
        return self.get_graph(init_state, lidar_data)
    
    def get_lidar_data(self, states: State, obstacles: Obstacle) -> Float[Array, "n_agent top_k_rays 2"]:
        """Get lidar data for each agent."""
        lidar_data = None
        # Use the static self.params["n_obs"] here.
        if self._params["n_obs"] > 0:
            get_lidar_vmap = jax_vmap(
                ft.partial(
                    get_lidar,
                    obstacles=obstacles,
                    # num_beams=self.n_rays,  # Using static attribute
                    num_beams=32,  # Using static attribute
                    sense_range=self.lidar_radius,  # Using static attribute
                    max_returns=self.top_k_rays,  # Using static attribute
                )
            )
            lidar_data = get_lidar_vmap(states[:, :2])
            # This assert now uses a static value for top_k_rays.
            assert lidar_data.shape == (self.num_agents, self.top_k_rays, 2)
        return lidar_data

    def step(
            self, graph: LidarEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[LidarEnvGraphsTuple, Reward, Cost, Done, Info]:
        env_state = graph.env_states
        action = self.clip_action(action)
        assert action.shape == (self.num_agents, 2)
        
        world = World(x_semidim=1.2, y_semidim=1.2, contact_margin=6e-3, substeps=5, collision_force=500)
        
        agents = [
            Agent.create(
                f"agent_{ii}",
                u_multiplier=0.5,
                shape=Sphere(self.agent_radius),
                collision_filter=lambda other: other.name == "object",
            ).withstate(
                pos=env_state.a_pos[ii], 
                vel=env_state.a_vel[ii]
            ).withforce(
                force=action[ii] * 0.5
            )
            for ii in range(self.num_agents)
        ]
        
        object = Entity.create(
            "object",
            movable=True,
            rotatable=True,
            collide=True,
            shape=Object(length=self.object_length),
            mass=self.object_mass,
        ).withstate(
            pos=env_state.object_pos[0],
            vel=env_state.object_vel[0],
            rot=env_state.object_angle[0],
            ang_vel=env_state.object_angvel[0]
        )

        entities, info = world.step([object, *agents])
        object = entities[0]
        agents = entities[1:]
        
        new_state = VMASCollaborativeTransportLidarState(
            agent=jnp.stack([
                jnp.concatenate([
                    agent.state.pos, 
                    agent.state.vel, 
                    jnp.zeros((2,), dtype=agent.state.pos.dtype)
                ])
                for agent in agents
            ]),
            object=jnp.concatenate([
                object.state.pos,
                object.state.vel,
                object.state.rot[0:1],
                object.state.ang_vel[0:1]
            ])[None, :],  # Wrap to get shape (1, 6)
            goal=env_state.goal,
            obstacle=env_state.obstacle
        )

        lidar_data_next = self.get_lidar_data(new_state.agent, new_state.obstacle)
        
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        done = jnp.array(False)
        info = {}
        self.step_count += 1
        
        return self.get_graph(new_state, lidar_data_next), reward, cost, done, info

    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        env_state: VMASCollaborativeTransportLidarState = graph.env_states
        object_pos = env_state.object_pos
        goal_pos = env_state.goal_pos
        agent_pos = env_state.a_pos

        angles = env_state.object_angle + jnp.array([0, 2*jnp.pi/3, 4*jnp.pi/3])
        vertex_pos = env_state.object_pos + self.object_radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        agent_vertex_dists = jnp.linalg.norm(agent_pos - vertex_pos, axis=-1)

        # dist2goal = jnp.linalg.norm(goal_pos - object_pos, axis=-1)
        dist2goal = jnp.linalg.norm(goal_pos - agent_pos, axis=-1)
        # jax.debug.print("dist2goal: {x}", x=dist2goal)
        reward = -dist2goal.mean() * 0.01
        reward -= jnp.where(dist2goal > self.goal_threshold, 1.0, 0.0).mean() * 0.001
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001
        reward -= agent_vertex_dists.sum() * 0.1
        reward -= jnp.abs(env_state.a_vel).sum() * 0.01

        return reward
    def get_cost(self, graph: GraphsTuple) -> Cost:
        env_state: VMASCollaborativeTransportLidarState = graph.env_states
        agent_pos = env_state.a_pos
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        a_cost_agent: Array = self.agent_radius * 2 - min_dist

        angles = env_state.object_angle + jnp.array([0, 2*jnp.pi/3, 4*jnp.pi/3])
        vertex_pos = env_state.object_pos + self.object_radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

        if self.params['n_obs'] == 0:
            obs_cost = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        else:
            obs_pos = graph.type_states(type_idx=2, n_type=self._params["top_k_rays"] * self.num_agents)[:, :2]
            obs_pos = jnp.reshape(obs_pos, (self.num_agents, self._params["top_k_rays"], 2))
            # Replace lidar readings that are (approximately) zero with a large constant.
            # Here, we define "large" as 10 times the arena size.
            large_val = self.area_size * 10.0
            mask = jnp.all(jnp.isclose(obs_pos, 0.0), axis=-1, keepdims=True)  # shape (n_agents, top_k_rays, 1)
            obs_pos = jnp.where(mask, large_val * jnp.ones_like(obs_pos), obs_pos)
            dist_obs = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)  # shape (n_agents, top_k_rays)
            obs_cost = self.agent_radius - dist_obs.min(axis=1)

        # Compute distances from the triangle edges to obstacle points
        edge_dists = []  # will hold per-edge min distances for each obstacle, shape (3, n_objects)
        # Get obstacle positions from the graph (assumed type index 2)
        obs_pos = graph.type_states(type_idx=2, n_type=self._params["top_k_rays"] * self.num_objects)[:, :2]
        # Reshape so that each obstacle has top_k_rays points
        obs_pos = jnp.reshape(obs_pos, (self.num_objects, self._params["top_k_rays"], 2))
        vertex_pos = jnp.squeeze(vertex_pos, axis=0)  # Now shape (3, 2)

        for i in range(3):
            # Get the two vertices defining the current edge
            v1 = vertex_pos[i]              # shape (2,)
            v2 = vertex_pos[(i + 1) % 3]      # shape (2,)
            
            # Compute the edge vector and its unit vector
            edge = v2 - v1                  # shape (2,)
            edge_length = jnp.linalg.norm(edge)
            edge_unit = edge / edge_length  # shape (2,)
            
            # Compute vector from v1 to each obstacle point (broadcasting v1)
            to_obstacle = obs_pos - v1       # shape (n_objects, top_k_rays, 2)
            
            # Project these vectors onto the edge unit vector
            proj_length = jnp.sum(to_obstacle * edge_unit, axis=-1)  # shape (n_objects, top_k_rays)
            proj_length = jnp.clip(proj_length, 0, edge_length)        # ensure the projection lies on the edge
            
            # Find the closest point on the edge for each obstacle point
            closest_point = v1 + proj_length[..., None] * edge_unit   # shape (n_objects, top_k_rays, 2)
            
            # Compute the distance from each obstacle point to its closest point on the edge
            dists = jnp.linalg.norm(obs_pos - closest_point, axis=-1)   # shape (n_objects, top_k_rays)
            
            # For each obstacle, take the minimum distance (across its lidar returns) for this edge
            min_dist_edge = jnp.min(dists, axis=-1)                     # shape (n_objects,)
            edge_dists.append(min_dist_edge)

        # Stack the per-edge results: shape becomes (3, n_objects)
        edge_dists = jnp.stack(edge_dists, axis=0)

        # For each obstacle, pick the smallest distance among the three edges
        min_edge_dist = jnp.min(edge_dists, axis=0)  # shape (n_objects,)
        # Compute obstacle cost from the difference between a reference radius and these distances.
        obstacle_object_cost = jnp.full((self.num_agents,), jnp.sum(self.agent_radius - min_edge_dist))

        cost = jnp.stack([4 * a_cost_agent, 2 * obs_cost, 2 * obstacle_object_cost], axis=1)
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)
        assert cost.shape == (self.num_agents, self.n_cost)

        return cost
    def state2feat(self, state: State) -> Array:
        return state
    def edge_blocks(self, state: VMASCollaborativeTransportLidarState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        # Agent-Agent edges (remain unchanged)
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        edge_feats = (jax_vmap(self.state2feat)(state.agent)[:, None, :] -
                    jax_vmap(self.state2feat)(state.agent)[None, :, :])
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(edge_feats, agent_agent_mask, id_agent, id_agent)

        # Agent-Goal edges (simplified version)
        agent_goal_edges = []
        # Here, we assume state.goal has shape (num_agents, state_dim)
        for i_agent in range(self.num_agents):
            # Compute full state difference between agent and its corresponding goal.
            diff = state.agent[i_agent] - state.goal[i_agent]
            # No extra padding is needed if self.edge_dim equals state_dim.
            agent_goal_edges.append(
                EdgeBlock(diff[None, None, :], jnp.ones((1, 1)),
                        jnp.array([i_agent]), jnp.array([i_agent + self.num_agents]))
            )

        # Agent-Obstacle (lidar) edges
        agent_obs_edges = []
        n_hits = self._params["top_k_rays"] * self.num_agents
        if lidar_data is not None:
            # Observations are appended after agents and goals in the node ordering.
            # id_obs = jnp.arange(self.num_agents + self.num_goals, self.num_agents + self.num_goals + n_hits)
            id_obs = jnp.arange(self.num_agents + self.num_goals + self.num_objects,
                    self.num_agents + self.num_goals + self.num_objects + n_hits)

            for i in range(self.num_agents):
                id_hits = jnp.arange(i * self._params["top_k_rays"], (i + 1) * self._params["top_k_rays"])
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                # Only consider a reading active if its distance is less than comm_radius - 0.1
                # AND greater than a small threshold (e.g., 1e-3) to filter out dummy zeros.
                active_lidar = jnp.logical_and(
                    jnp.less(lidar_dist, self._params["lidar_radius"] - 1e-1),
                    jnp.greater(lidar_dist, 1e-3)
                )
                agent_obs_mask = jnp.ones((1, self._params["top_k_rays"]))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                # If edge_dim > 2, pad the lidar_feats with zeros.
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((lidar_feats.shape[0], self.edge_dim - lidar_feats.shape[1]))],
                    axis=-1
                )
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )

            # jax.debug.print("agent_obs_edges: {x}", x=agent_obs_edges)

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges


    def edge_blocks_local(self, state: VMASCollaborativeTransportLidarState, lidar_data: Optional[jnp.ndarray] = None) -> list:
        edge_blocks = []
        for i in range(self.num_agents):
            agent_goal_feats = jnp.concatenate([
                state.a_pos[i] - state.goal_pos,
                state.a_vel[i]
            ])[None, None, :]
            agent_goal_edges = EdgeBlock(
                agent_goal_feats, 
                jnp.ones((1, 1)), 
                jnp.array([i]), 
                jnp.array([self.num_agents])
            )
            edge_blocks.append(agent_goal_edges)
        
        if lidar_data is not None:
            n_hits = lidar_data.shape[0]
            if n_hits > 0:
                hits_per_agent = self.top_k_rays
                for i in range(self.num_agents):
                    start_idx = i * hits_per_agent
                    end_idx = min((i + 1) * hits_per_agent, n_hits)
                    if start_idx < n_hits:
                        agent_hits = lidar_data[start_idx:end_idx]
                        n_agent_hits = agent_hits.shape[0]
                        rel_positions = agent_hits - state.a_pos[i]
                        lidar_edge_feats = jnp.concatenate([
                            rel_positions,
                            jnp.zeros((n_agent_hits, 2))
                        ], axis=-1)
                        lidar_indices = jnp.arange(
                            self.num_agents + self.num_goals + start_idx,
                            self.num_agents + self.num_goals + end_idx
                        )
                        agent_lidar_edges = EdgeBlock(
                            lidar_edge_feats[None, :, :],
                            jnp.ones((1, n_agent_hits)),
                            jnp.array([i]),
                            lidar_indices
                        )
                        edge_blocks.append(agent_lidar_edges)
        
        return edge_blocks

    def get_graph(self, state: VMASCollaborativeTransportLidarState, lidar_data: Pos2d = None) -> GraphsTuple:
        """Create a graph representation of the environment state."""
        # Use the static self.params["n_obs"] to determine lidar hits.
        n_hits = self.top_k_rays * self.num_agents if self._params["n_obs"] > 0 else 0
        n_nodes = self.num_agents + self.num_goals + self.num_objects + n_hits

        # Process lidar data if not provided.
        if lidar_data is not None:
            lidar_data = merge01(lidar_data)

        # object_dim = state_dim = 6
        node_feats = jnp.zeros((self.num_agents + self.num_goals + self.num_objects + n_hits, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, :self.state_dim].set(state.agent)
        # node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, :2].set(state.goal)
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, :2].set(state.goal[:, :2])

        node_feats = node_feats.at[self.num_agents + self.num_goals: self.num_agents + self.num_goals + self.num_objects, :self.object_dim].set(state.object)
        if lidar_data is not None:
            node_feats = node_feats.at[-n_hits:, :2].set(lidar_data)
            
        # indicators
        node_feats = node_feats.at[: self.num_agents, self.state_dim + 3].set(1.)  # agent
        node_feats = (
            node_feats.at[self.num_agents: self.num_agents + self.num_goals, self.state_dim + 2].set(1.))  # goal
        node_feats = node_feats.at[self.num_agents + self.num_goals: self.num_agents + self.num_goals + self.num_objects, self.state_dim + 1].set(1.)  # object
        if n_hits > 0:
            node_feats = node_feats.at[-n_hits:, self.state_dim].set(1.)  # obs feats


        # Create node type one-hot encodings
        # node_type = jnp.zeros((n_nodes, 4))
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[: self.num_agents].set(VMASCollaborativeTransportLidar.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(VMASCollaborativeTransportLidar.GOAL)
        node_type = node_type.at[self.num_agents + self.num_goals:self.num_agents + self.num_goals + self.num_objects].set(VMASCollaborativeTransportLidar.OBJECT)
        if n_hits > 0:
            node_type = node_type.at[-n_hits:].set(VMASCollaborativeTransportLidar.OBS)

        # node_type = node_type.at[:self.num_agents, self.AGENT].set(1.0)
        # node_type = node_type.at[self.num_agents:self.num_agents + self.num_goals, self.GOAL].set(1.0)
        # node_type = node_type.at[self.num_agents + self.num_goals:self.num_agents + self.num_goals + self.num_objects, self.OBJECT].set(1.0)
        # if n_hits > 0:
            # node_type = node_type.at[-n_hits:, self.OBS].set(1.0)

        if self.local_only:
            edge_blocks = self.edge_blocks_local(state, lidar_data)
        else:
            edge_blocks = self.edge_blocks(state, lidar_data)
            
        states = jnp.concatenate([state.agent, state.goal, state.object], axis=0)
        if lidar_data is not None:
            lidar_states = jnp.concatenate(
                [lidar_data, jnp.zeros((n_hits, self.state_dim - lidar_data.shape[1]))], axis=1)
            states = jnp.concatenate([states, lidar_states], axis=0)
            
        return GetGraph(nodes=node_feats, node_type=node_type, edge_blocks=edge_blocks, states=states, env_states=state).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        pass

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim
    def get_obs_collection(self, obstacles: Obstacle, color: str, alpha: float):
        if isinstance(obstacles, Rectangle):
            n_obs = len(obstacles.center)
            obs_polys = [Polygon(obstacles.points[ii]) for ii in range(n_obs)]
            obs_col = PatchCollection(obs_polys, color=color, alpha=alpha, zorder=99)
        else:
            raise NotImplementedError
        return obs_col

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            n_goal: int = None,
            dpi: int = 200,
            **kwargs,
    ) -> None:
        T_graph = rollout.graph
        T_env_states: VMASCollaborativeTransportLidarState = T_graph.env_states
        T_costs = rollout.costs
        graph0 = tree_index(T_graph, 0)
        obs_color = "#8a0000"
        goal_color = "#2fdd00"
        edge_goal_color = goal_color
        edge_obs_color = obs_color
        goals = graph0.type_states(type_idx=1, n_type=self.num_goals)
        goal_pos_list = np.array(goals[:, :2])

        # Define node counts according to the ordering in get_graph:
        # agents, goals, objects, then lidar (obstacle) nodes.
        n_agent = self.num_agents
        n_goal = self.num_goals
        n_object = self.num_objects
        n_hits = self.top_k_rays * self.num_agents if self._params["n_obs"] > 0 else 0
        total_nodes = n_agent + n_goal + n_object + n_hits

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
        ax.set_xlim(-1.01 * self.area_size, 1.01 * self.area_size)
        ax.set_ylim(-1.01 * self.area_size, 1.01 * self.area_size)
        ax.set_aspect("equal")

        # Draw the arena boundaries.
        ax.add_patch(
            plt.Rectangle(
                (-self.area_size, -self.area_size),
                2 * self.area_size,
                2 * self.area_size,
                fc="none",
                ec="C3",
            )
        )

        dist2goal = self.goal_threshold        
        # Ensure goal_pos is a 2D point.
        # Plot all three goal positions
        goal_colors = ["C5", "C5", "C5"]  # Different colors for each goal
        for i in range(3):  # There are 3 goal positions
            goal_pos = goal_pos_list[i]
            goal_circ = plt.Circle((goal_pos[0], goal_pos[1]), dist2goal, 
                                  color=goal_colors[i % len(goal_colors)], alpha=0.5)
            ax.add_patch(goal_circ)

        # Plot obstacles if available.
        if hasattr(graph0.env_states, "obstacle"):
            obs = graph0.env_states.obstacle
            if obs is not None:
                ax.add_collection(self.get_obs_collection(obs, obs_color, alpha=0.8))

        # Create the triangle patch (for the object).
        base_angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
        triangle_vertices = self.object_radius * np.array([
            [np.cos(angle), np.sin(angle)] for angle in base_angles
        ])
        triangle_patch = plt.Polygon(triangle_vertices, ec="C3", fc="none")
        ax.add_patch(triangle_patch)

        # object_center = plt.Circle((0, 0), 0.5 * dist2goal, fc="C3", ec="none", zorder=6)
        # ax.add_patch(object_center)

        # Create agent patches.
        agent_colors = ["C2", "C1", "C4"]
        agent_radius = self.agent_radius
        agent_patches = [
            plt.Circle((0, 0), agent_radius, color=agent_colors[ii], zorder=5)
            for ii in range(n_agent)
        ]
        for patch in agent_patches:
            ax.add_patch(patch)

        # Texts for debug.
        text_font_opts = dict(
            size=16,
            color="k",
            family="cursive",
            weight="normal",
            transform=ax.transAxes,
        )
        goal_text = ax.text(0.99, 1.00, "dist_goal=0", va="bottom", ha="right", **text_font_opts)
        obs_text = ax.text(0.99, 1.04, "dist_obs=0", va="bottom", ha="right", **text_font_opts)
        cost_text = ax.text(0.99, 1.12, "cost=0", va="bottom", ha="right", **text_font_opts)
        kk_text = ax.text(0.99, 1.08, "kk=0", va="bottom", ha="right", **text_font_opts)
        texts = [goal_text, obs_text, kk_text, cost_text]

        # For drawing edges, we use a LineCollection.
        from matplotlib.collections import LineCollection
        edge_col = LineCollection([], colors=[], linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(edge_col)

        # Define dimension (2D environment)
        dim = 2

        # Debug prints (optional)
        # jax.debug.print("n_agent: {x}", x=n_agent)
        # jax.debug.print("n_goal: {x}", x=n_goal)
        # jax.debug.print("n_hits: {x}", x=n_hits)
        # jax.debug.print("total_nodes: {x}", x=total_nodes)

        # Initial edge drawing using the full node set.
        all_pos = np.array(graph0.states[:total_nodes, :dim])
        if dim == 1:
            all_pos = np.concatenate([all_pos, np.ones((total_nodes, 1)) * self.area_size / 2], axis=1)
        edge_index = np.stack([np.array(graph0.senders), np.array(graph0.receivers)], axis=0)
        # Assume padding uses the index equal to total_nodes.
        is_pad = np.any(edge_index == total_nodes, axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        # jax.debug.print("all_pos shape: {x}", x=all_pos.shape)
        # jax.debug.print("e_edge_index shape: {x}", x=e_edge_index.shape)
        # jax.debug.print("e_edge_index max value: {x}", x=e_edge_index.max())
        e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        e_lines = np.stack([e_start, e_end], axis=1)  # (num_edges, 2, dim)
        # For this example, we color an edge differently if its sender is a goal node.
        e_is_goal = (e_edge_index[0, :] >= n_agent) & (e_edge_index[0, :] < n_agent + n_goal)
        e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(e_lines.shape[0])]
        if dim == 1:
            edge_col = LineCollection(e_lines[~e_is_goal], colors="0.2", linewidths=2, alpha=0.5, zorder=3)
        elif dim == 2:
            edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        else:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            edge_col = Line3DCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(edge_col)

        def init_fn() -> list[plt.Artist]:
            # Return all static artists including the edge collection.
            return [triangle_patch,  *agent_patches, *texts, edge_col]

        def update(kk: int) -> list[plt.Artist]:
            # Get the current environment state and graph.
            env_state: VMASCollaborativeTransportLidarState = tree_index(T_env_states, kk)
            cur_graph = tree_index(T_graph, kk)
            
            # Update agent positions.
            for ii in range(n_agent):
                pos = env_state.a_pos[ii]
                agent_patches[ii].set_center(pos)
            
            # Update the object (triangle) patch.
            angle = float(np.array(env_state.object_angle)[0, 0])
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            rotated_vertices = np.dot(triangle_vertices, rotation_matrix.T)
            obj_center_coord = np.array(env_state.object_pos)[0]
            transformed_vertices = rotated_vertices + obj_center_coord
            triangle_patch.set_xy(transformed_vertices)
            # object_center.set_center(obj_center_coord)
            
            # Update debug texts.
            o_dist_obs = np.linalg.norm(env_state.object_pos - env_state.obstacle[0], axis=-1) - self.object_radius
            dist_obs_str = ", ".join(["{:+.3f}".format(d) for d in o_dist_obs])
            dist_goal = np.linalg.norm(env_state.object_pos - env_state.goal_pos)
            cost_str = ", ".join(["{:+.3f}".format(c) for c in T_costs[kk].max(0)])
            obs_text.set_text("dist_obs=[{}]".format(dist_obs_str))
            goal_text.set_text("dist_goal={:.3f}".format(dist_goal))
            kk_text.set_text("kk={:04}".format(kk))
            cost_text.set_text("cost={}".format(cost_str))
            
            # ----- Update edges -----
            # Use the full node array (agents, goals, objects, lidar nodes).
            all_pos = np.array(cur_graph.states[:total_nodes, :dim])
            edge_index = np.stack([np.array(cur_graph.senders), np.array(cur_graph.receivers)], axis=0)
            is_pad = np.any(edge_index == total_nodes, axis=0)
            e_edge_index = edge_index[:, ~is_pad]
            if e_edge_index.shape[1] > 0:
                e_start = all_pos[e_edge_index[0, :]]
                e_end = all_pos[e_edge_index[1, :]]
                
                e_lines = np.stack([e_start, e_end], axis=1)
                e_is_goal = (e_edge_index[0, :] >= n_agent) & (e_edge_index[0, :] < n_agent + n_goal)
                e_is_obs = (e_edge_index[0, :] >= n_agent+ n_goal)
                e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(e_lines.shape[0])]
                e_colors = [edge_obs_color if e_is_obs[ii] else "0.2" for ii in range(e_lines.shape[0])]
                # for e_is_obs_element, jj in enumerate(e_is_obs):
                #     if e_is_obs_element and len(e_start[jj]) > 0 :
                #         jax.debug.print("e_edge_index0: {x}", x=e_edge_index[0, :])
                #         jax.debug.print("e_edge_index1: {x}", x=e_edge_index[1, :])
                #         jax.debug.print("e_start: {x}", x=len(e_start[jj]))
                #         jax.debug.print("e_start: {x}", x=e_start[jj])
                #         jax.debug.print("e_end: {x}", x=e_end[jj])
            else:
                e_lines = []
                e_colors = []
            edge_col.set_segments(e_lines)
            edge_col.set_color(e_colors)
            # -------------------------

            return [triangle_patch, *agent_patches, *texts, edge_col]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)
