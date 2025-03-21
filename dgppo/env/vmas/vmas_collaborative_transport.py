import pathlib
import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from typing import NamedTuple, Optional, Tuple, Dict
from matplotlib.animation import FuncAnimation

from .physax.entity import Agent, Entity
from .physax.shapes import Box, Sphere, Object
from .physax.world import World
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from dgppo.utils.typing import Action, Array, Cost, Done, Info, Reward, State
from dgppo.utils.utils import save_anim, tree_index
from dgppo.env.base import MultiAgentEnv
from dgppo.env.utils import get_node_goal_rng



class VMASCollaborativeTransportState(NamedTuple):
    object_pos: Array
    object_vel: Array
    object_angle: Array
    object_angvel: Array
    a_pos: Array
    a_vel: Array
    goal_pos: Array
    o_pos: Array


class VMASCollaborativeTransport(MultiAgentEnv):
    AGENT = 0

    PARAMS = {
        "comm_radius": 0.4,
        "default_area_size": 0.8,
        "dist2goal": 0.01,
        "agent_radius": 0.03,
        "object_length": 0.1,
        "object_mass": 10.0
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 64,
            dt: float = 0.1,
            params: dict = None,
            object_length: float = 0.1,
            object_mass: float = 10.0,
            half_width: float = 0.8
    ):
        self.object_length = object_length
        self.object_mass = object_mass
        self.half_width = half_width
        assert num_agents == 3, "VMASCollaborativeTransport only supports 3 agents."
        area_size = 2 * half_width
        self.agent_radius = 0.03
        super().__init__(3, area_size, max_step, dt, params)

        self.obs_radius = 0.15
        self.n_obs = 3

        self.frame_skip = 4

        # Initialize spaces with concrete types, not Protocols
        self.observation_space = {
            'shape': (num_agents, self.n_obs),
            'dtype': jnp.float32
        }
        self.action_space = {
            'shape': (num_agents, 2),
            'dtype': jnp.float32
        }

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        # [pos(2), vel(2), box_pos(2), box_vel(2), box_angle(1), box_angvel(1), rel_goal_pos(2), in_contact(1), rel_obs_pos_vec(6), rel_obs_dist(3)]
        return 21

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # fx, fy

    @property
    def n_cost(self) -> int:
        return 3

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obstacle collisions"

    def reset(self, key: Array) -> GraphsTuple:
        object_key, goal_key, obs_key = jax.random.split(key, 3)
        
        # Sample object position and orientation
        obj_pos_key, obj_angle_key = jax.random.split(object_key, 2)

        # box_cen_halfwidth = self.half_width - 0.5 * self.package_length

         # Sample object position within valid radius
        obj_cen_halfwidth = self.half_width - self.object_length
        obj_radius = 0.98 * obj_cen_halfwidth
        obj_pos_angle = jax.random.uniform(obj_pos_key, minval=0.0, maxval=2 * np.pi)
        obj_pos = obj_radius * jnp.array([jnp.cos(obj_pos_angle), jnp.sin(obj_pos_angle)])
        
        # Sample object rotation angle
        obj_angle = jax.random.uniform(obj_angle_key, minval=0.0, maxval=2 * np.pi)
        
        # Calculate agent positions at triangle vertices
        angles = jnp.array([obj_angle + i * 2 * jnp.pi / 3 for i in range(3)])
        agent_pos = obj_pos + self.object_length * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        
        # Initialize velocities as zero
        obj_vel = jnp.zeros(2)
        obj_angvel = jnp.array(0.0)
        agent_vel = jnp.zeros((self.num_agents, 2))

        # Sample goal position opposite to object
        goal_radius = obj_radius
        noise_ub = np.deg2rad(30)
        goal_angle = obj_pos_angle + np.pi + jax.random.uniform(goal_key, minval=-noise_ub, maxval=noise_ub)
        goal_pos = goal_radius * jnp.array([jnp.cos(goal_angle), jnp.sin(goal_angle)])

        # Sample obstacle positions
        obs_radius = obj_radius - 1.5 * self.obs_radius
        o_angle = jax.random.uniform(obs_key, shape=(self.n_obs,), minval=0.0, maxval=2 * np.pi)
        o_pos = obs_radius * jnp.stack([jnp.cos(o_angle), jnp.sin(o_angle)], axis=-1)

        env_state = VMASCollaborativeTransportState(
            obj_pos, obj_vel, obj_angle, obj_angvel, agent_pos, agent_vel, goal_pos, o_pos
        )
        return self.get_graph(env_state)

    def step(
            self, graph: GraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[GraphsTuple, Reward, Cost, Done, Info]:
        action = self.clip_action(action)
        assert action.shape == (self.num_agents, 2)

        env_state: VMASCollaborativeTransportState = graph.env_states

        # Create world instance
        world = World(x_semidim=1.2, y_semidim=1.2, contact_margin=6e-3, substeps=5, collision_force=500)

        # Create and update entities
        agents = [
            Agent.create(
                f"agent_{ii}",
                u_multiplier=0.5,
                shape=Sphere(self.agent_radius),
                collision_filter=lambda other: other.name == "object",
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
        )

        # Update states and forces
        for ii, agent in enumerate(agents):
            agent = agent.withstate(pos=env_state.a_pos[ii], vel=env_state.a_vel[ii])
            agent = agent.withforce(force=action[ii] * agent.u_multiplier)
            agents[ii] = agent
        
        object = object.withstate(
            pos=env_state.object_pos, 
            vel=env_state.object_vel, 
            rot=jnp.array([env_state.object_angle]), 
            ang_vel=jnp.array([env_state.object_angvel])
        )

        entities = [object, *agents]
        
        # Run physics simulation
        if self.frame_skip > 1:
            def body(entities_, _):
                entities_, _ = world.step(entities_)
                return entities_, None
            entities_secondlast, _ = lax.scan(body, entities, None, length=self.frame_skip - 1)
        else:
            entities_secondlast = entities

        # Final step
        entities, info = world.step(entities_secondlast)
        
        # Extract updated states
        object = entities[0]  # First entity is the object
        agents = entities[1:]  # Rest are agents
        
        # Create new environment state with updated positions
        env_state_new = VMASCollaborativeTransportState(
            object_pos=object.state.pos,
            object_vel=object.state.vel,
            object_angle=object.state.rot[0],
            object_angvel=object.state.ang_vel[0],
            a_pos=jnp.stack([agent.state.pos for agent in agents]),
            a_vel=jnp.stack([agent.state.vel for agent in agents]),
            o_pos=env_state.o_pos,  # Obstacles don't move
            goal_pos=env_state.goal_pos  # Goal doesn't move
        )

        # Calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        done = jnp.array(False)

        return self.get_graph(env_state_new), reward, cost, done, info

    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        env_state: VMASCollaborativeTransportState = graph.env_states

        object_pos = env_state.object_pos
        goal_pos = env_state.goal_pos
        agent_pos = env_state.a_pos

        # Calculate triangle vertices
        angles = env_state.object_angle + jnp.array([0, 2*jnp.pi/3, 4*jnp.pi/3])
        vertex_pos = env_state.object_pos + self.object_length * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        # vertex_pos shape: (3, 2) for three vertices
        
        # Calculate distances from agents to their corresponding vertices
        agent_vertex_dists = jnp.linalg.norm(agent_pos - vertex_pos, axis=-1)  # shape: (3,)

        # goal distance penalty
        dist2goal = jnp.linalg.norm(goal_pos - object_pos, axis=-1)
        reward = -dist2goal.mean() * 0.01

        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001
        
        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        # Add vertex deviation penalty to total reward
        reward -= agent_vertex_dists.sum() * 0.1
        
        reward -= jnp.abs(env_state.a_vel).sum() * 0.01  # Penalty for absolute agent velocities
        
        # Penalty for object rotation speed
        reward += jnp.abs(env_state.object_angvel) * 0.05  # Penalize fast rotation

        return reward

    def get_cost(self, graph: GraphsTuple) -> Cost:
        env_state: VMASCollaborativeTransportState = graph.env_states
        agent_pos = env_state.a_pos

        # collision between agents
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        # (n_agent, )
        a_cost_agent: Array = self.params["agent_radius"] * 2 - min_dist

        # Calculate triangle vertices
        angles = env_state.object_angle + jnp.array([0, 2*jnp.pi/3, 4*jnp.pi/3])
        vertex_pos = env_state.object_pos + self.object_length * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        # vertex_pos shape: (3, 2) for three vertices

        # Calculate distances from obstacles to each edge of the triangle
        edge_dists = []
        for i in range(3):
            # Get two vertices that form this edge
            v1 = vertex_pos[i]
            v2 = vertex_pos[(i + 1) % 3]
            
            # Vector along the edge
            edge = v2 - v1
            edge_length = jnp.linalg.norm(edge)
            edge_unit = edge / edge_length
            
            # Vector from edge start to obstacles
            to_obstacle = env_state.o_pos - v1[None, :]  # shape: (n_obs, 2)
            
            # Project onto edge vector
            proj_length = jnp.sum(to_obstacle * edge_unit, axis=-1)  # shape: (n_obs,)
            
            # Clamp projection to edge length
            proj_length = jnp.clip(proj_length, 0, edge_length)
            
            # Closest point on edge
            closest_point = v1[None, :] + proj_length[:, None] * edge_unit
            
            # Calculate absolute distance from obstacle to closest point
            dist = jnp.abs(jnp.linalg.norm(env_state.o_pos - closest_point, axis=-1))
            edge_dists.append(dist)

        # Stack distances from all edges
        edge_dists = jnp.stack(edge_dists)  # shape: (3 edges, n_obs)

        # Find minimum absolute distance for each obstacle to any edge
        min_edge_dist = jnp.min(edge_dists, axis=0)  # shape: (n_obs,)

        # Calculate total cost as sum of costs for each obstacle
        total_cost = jnp.sum(self.obs_radius - min_edge_dist)
        # Broadcast total_cost to match shape of a_cost_agent
        total_cost_object_obstacle = jnp.full_like(a_cost_agent, total_cost)

        # Calculate distances from agents to obstacles
        agent_to_obs = jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(env_state.o_pos, 0)  # shape: (n_agent, n_obs, 2)
        agent_obs_dist = jnp.linalg.norm(agent_to_obs, axis=-1)  # shape: (n_agent, n_obs)
        a_cost_obs = (self.params["agent_radius"] + self.obs_radius) - agent_obs_dist  # shape: (n_agent, n_obs)
        a_cost_obs = jnp.max(a_cost_obs, axis=-1)  # shape: (n_agent,)

        # Stack costs and reshape
        cost = jnp.stack([4 * a_cost_agent,  2 * a_cost_obs, 2 * total_cost_object_obstacle], axis=1)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)
        assert cost.shape == (self.num_agents, self.n_cost)

        return cost

    def get_graph(self, env_state: VMASCollaborativeTransportState) -> GraphsTuple:
        state = env_state

        rel_goal_pos = state.goal_pos - state.object_pos
        
        o_rel_obspos = state.o_pos - state.object_pos
        assert o_rel_obspos.shape == (self.n_obs, 2)
        o_dist = jnp.sqrt(jnp.sum(o_rel_obspos ** 2, axis=-1) + 1e-6)
        o_rel_obspos_vec = o_rel_obspos / o_dist[:, None]

        idx_sort = jnp.argsort(o_dist)
        o_rel_obspos_vec = o_rel_obspos_vec[idx_sort]
        o_dist = o_dist[idx_sort]

        # node features.
        node_feats = jnp.zeros((self.num_agents, self.node_dim))
        node_feats = node_feats.at[:, :2].set(state.a_pos)
        node_feats = node_feats.at[:, 2:4].set(state.a_vel)
        node_feats = node_feats.at[:, 4:6].set(state.object_pos)
        node_feats = node_feats.at[:, 6:8].set(state.object_vel)
        node_feats = node_feats.at[:, 8].set(state.object_angle)
        node_feats = node_feats.at[:, 9].set(state.object_angvel)
        node_feats = node_feats.at[:, 10:12].set(rel_goal_pos)
        node_feats = node_feats.at[:, 12:18].set(o_rel_obspos_vec.flatten())
        node_feats = node_feats.at[:, 18:21].set(o_dist)

        node_type = jnp.full(self.num_agents, VMASCollaborativeTransport.AGENT)
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        n_state_vec = jnp.zeros((self.num_agents, 0))
        return GetGraph(node_feats, node_type, edge_blocks, env_state, n_state_vec).to_padded()

    def edge_blocks(self, env_state: VMASCollaborativeTransportState) -> list[EdgeBlock]:
        state = env_state

        nagent = self.num_agents
        agent_pos = state.a_pos
        agent_vel = state.a_vel
        agent_states = jnp.concatenate([agent_pos[:, :2], agent_vel[:, :2]], axis=-1)

        # agent - agent connection
        state_diff = agent_states[:, None, :] - agent_states[None, :, :]
        agent_agent_mask = jnp.array(jnp.eye(nagent) == 0)
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        return [agent_agent_edges]

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        pass

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

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
        T_env_states: VMASCollaborativeTransportState = T_graph.env_states
        T_costs = rollout.costs

        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
        ax.set_xlim(-1.01 * self.half_width, 1.01 * self.half_width)
        ax.set_ylim(-1.01 * self.half_width, 1.01 * self.half_width)
        ax.set_aspect("equal")

        # Plot a rectangle to visualize the halfwidth
        ax.add_patch(
            plt.Rectangle(
                (-self.half_width, -self.half_width), 2 * self.half_width, 2 * self.half_width, fc="none", ec="C3"
            )
        )

        # Plot a circle for the goal.
        goal_pos = T_env_states.goal_pos[0]
        dist2goal = self.params["dist2goal"]
        goal_circ = plt.Circle(goal_pos, dist2goal, color="C5", alpha=0.5)
        ax.add_patch(goal_circ)

        # Plot the obstacles.
        o_pos = T_env_states.o_pos[0]
        for oo in range(self.n_obs):
            obs_circ = plt.Circle(o_pos[oo], self.obs_radius, fc="C0", ec="none", alpha=0.7)
            ax.add_patch(obs_circ)

        # Create base equilateral triangle vertices (before rotation)
        base_angles = np.array([0, 2*np.pi/3, 4*np.pi/3])  # Three angles 120 degrees apart
        triangle_vertices = self.object_length * np.array([
            [np.cos(angle), np.sin(angle)] for angle in base_angles
        ])
        triangle_patch = plt.Polygon(triangle_vertices, ec="C3", fc="none")
        ax.add_patch(triangle_patch)

        # Plot the center of the triangle
        object_center = plt.Circle((0, 0), 0.5 * dist2goal, fc="C3", ec="none", zorder=6)
        ax.add_patch(object_center)

        # Plot agent
        agent_colors = ["C2", "C1", "C4"]
        agent_radius = self.agent_radius
        agent_patches = [
            plt.Circle((0, 0), agent_radius, color=agent_colors[ii], zorder=5) for ii in range(self.num_agents)
        ]
        [ax.add_patch(patch) for patch in agent_patches]

        text_font_opts = dict(
            size=16,
            color="k",
            family="cursive",
            weight="normal",
            transform=ax.transAxes,
        )

        # text for line velocity
        goal_text = ax.text(0.99, 1.00, "dist_goal=0", va="bottom", ha="right", **text_font_opts)
        obs_text = ax.text(0.99, 1.04, "dist_obs=0", va="bottom", ha="right", **text_font_opts)
        cost_text = ax.text(0.99, 1.12, "cost=0", va="bottom", ha="right", **text_font_opts)

        # text for time step
        kk_text = ax.text(0.99, 1.08, "kk=0", va="bottom", ha="right", **text_font_opts)

        texts = [goal_text, obs_text, kk_text, cost_text]

        def init_fn() -> list[plt.Artist]:
            return [triangle_patch, object_center, *agent_patches, *texts]

        def update(kk: int) -> list[plt.Artist]:
            env_state: VMASCollaborativeTransportState = tree_index(T_env_states, kk)

            # update agent positions
            for ii in range(self.num_agents):
                pos = env_state.a_pos[ii]
                assert pos.shape == (2,)
                agent_patches[ii].set_center(pos)

            # Update triangle position and rotation
            # Rotate vertices by current object angle
            rotation_matrix = np.array([
                [np.cos(env_state.object_angle), -np.sin(env_state.object_angle)],
                [np.sin(env_state.object_angle), np.cos(env_state.object_angle)]
            ])
            rotated_vertices = np.dot(triangle_vertices, rotation_matrix.T)
            # Translate to current position
            transformed_vertices = rotated_vertices + env_state.object_pos
            triangle_patch.set_xy(transformed_vertices)
            object_center.set_center(env_state.object_pos)

            o_dist_obs = np.linalg.norm(env_state.object_pos - env_state.o_pos, axis=-1) - self.obs_radius
            dist_obs_str = ", ".join(["{:+.3f}".format(d) for d in o_dist_obs])
            dist_goal = np.linalg.norm(env_state.object_pos - env_state.goal_pos)

            cost_str = ", ".join(["{:+.3f}".format(c) for c in T_costs[kk].max(0)])

            obs_text.set_text("dist_obs=[{}]".format(dist_obs_str))
            goal_text.set_text("dist_goal={:.3f}".format(dist_goal))
            kk_text.set_text("kk={:04}".format(kk))
            cost_text.set_text("cost={}".format(cost_str))

            return [triangle_patch, object_center, *agent_patches, *texts]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)
