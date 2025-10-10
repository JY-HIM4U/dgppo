import pathlib
import pickle
import sys
import types
import importlib
import numpy as np


def make_exper_video_from_file(pickle_path: str, max_frames: int | None = None):
    """Load experiment data from pickle and render a video.

    The pickle is expected to contain a VideoData instance
    saved by DGPPO.save_for_video.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm

    path = pathlib.Path(pickle_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {pickle_path}")

    # Ensure project roots are on sys.path for imports during unpickling
    try:
        this_file = pathlib.Path(__file__).resolve()
        for parent in this_file.parents:
            if parent.name in {"src", "build"}:
                pstr = str(parent)
                if pstr not in sys.path:
                    sys.path.insert(0, pstr)
    except Exception:
        pass

    # Provide a compatibility shim for pickled class references
    # Some runs may have pickled using module name 'gcbf_ros2.dgppo_runner.VideoData'
    # while current code lives under 'gcbf_ros2.gcbf_ros2.dgppo_runner.VideoData'.
    legacy_mod = 'gcbf_ros2.dgppo_runner'
    # Ensure root package exists so submodule registration works
    if 'gcbf_ros2' not in sys.modules:
        try:
            import importlib as _importlib
            sys.modules['gcbf_ros2'] = _importlib.import_module('gcbf_ros2')
        except Exception:
            sys.modules['gcbf_ros2'] = types.ModuleType('gcbf_ros2')
    if legacy_mod not in sys.modules:
        mod = types.ModuleType(legacy_mod)
        class VideoData:  # minimal stub for unpickling
            def __init__(self, graphs, actions, costs, rewards, meta=None):
                self.graphs = graphs
                self.actions = actions
                self.costs = costs
                self.rewards = rewards
                self.meta = meta or {}
        mod.VideoData = VideoData
        sys.modules[legacy_mod] = mod

    # Shim legacy top-level package name 'model_dgppo' to current location 'gcbf_ros2.model_dgppo'
    try:
        current_pkg = importlib.import_module('gcbf_ros2.model_dgppo')
        sys.modules.setdefault('model_dgppo', current_pkg)
        # Common submodules used in pickles
        for sub in [
            'env',
            'env.vmas_lidar',
            'env.vmas_lidar.vmas_collaborative_transport_lidar',
            'env.obstacle',
            'utils',
        ]:
            try:
                real_name = f'gcbf_ros2.model_dgppo.{sub}'
                legacy_name = f'model_dgppo.{sub}'
                mod = importlib.import_module(real_name)
                sys.modules.setdefault(legacy_name, mod)
            except Exception:
                pass
    except Exception:
        pass

    # Load data: support both pickle (.pkl) and torch (.pth)
    video_data = None
    suffix = path.suffix.lower()
    try:
        if suffix in {".pkl", ".pickle"}:
            with path.open("rb") as f:
                video_data = pickle.load(f)
        elif suffix == ".pth":
            try:
                import torch  # type: ignore
                video_data = torch.load(str(path), map_location="cpu")
            except Exception:
                # Fallback to pickle if torch load fails
                with path.open("rb") as f:
                    video_data = pickle.load(f)
        else:
            # Try pickle first, then torch
            try:
                with path.open("rb") as f:
                    video_data = pickle.load(f)
            except Exception:
                import torch  # type: ignore
                video_data = torch.load(str(path), map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {pickle_path}: {e}")

    graph_list = video_data.graphs
    action_list = video_data.actions
    cost_list = video_data.costs
    rewards_list = video_data.rewards
    meta = getattr(video_data, "meta", {})

    if not graph_list:
        print("No data to create video from")
        return

    # Parameters
    agent_radius = meta.get("agent_radius", 0.1)
    area_size = meta.get("area_size", 2.0)
    polygon_length = meta.get("polygon_length", 0.3)
    num_agents = meta.get("num_agents", 3)
    num_goals = meta.get("num_goals", 1)
    num_objects = meta.get("num_objects", 1)
    top_k_rays = meta.get("top_k_rays", 0)
    n_obs = meta.get("n_obs", 0)

    first_graph = graph_list[0]
    env_state = first_graph.env_states

    n_agent = num_agents
    n_goal = num_goals
    n_object = num_objects
    n_hits = top_k_rays * num_agents if n_obs > 0 else 0
    total_nodes = n_agent + n_goal + n_object + n_hits

    num_agents_init = int(env_state.real_num_agents)
    object_length = polygon_length / (2 * np.sin(np.pi / num_agents_init))

    # Create figure with subplots: main view and trajectory plot
    fig = plt.figure(figsize=(15, 8), dpi=200)
    ax_main = plt.subplot(1, 2, 1)
    ax_traj = plt.subplot(1, 2, 2)
    
    # Main plot setup
    ax_main.set_xlim(-agent_radius, area_size + agent_radius)
    ax_main.set_ylim(-agent_radius, area_size + agent_radius)
    ax_main.set_aspect("equal")
    ax_main.set_title("Agent Positions", fontsize=14)
    
    # Trajectory plot setup
    ax_traj.set_xlim(0, len(graph_list))
    ax_traj.set_ylim(-agent_radius, area_size + agent_radius)
    ax_traj.set_xlabel("Time Step")
    ax_traj.set_ylabel("X Position")
    ax_traj.set_title("Agent X-Position Trajectories", fontsize=14)
    ax_traj.grid(True, alpha=0.3)

    ax_main.add_patch(
        plt.Rectangle(
            (-agent_radius, -agent_radius),
            area_size + 2 * agent_radius,
            area_size + 2 * agent_radius,
            fc="none",
            ec="C3",
        )
    )

    goal_pos = env_state.goal_pos[0]
    goal_theta = env_state.goal_theta[0, 0]
    base_angles = np.array([i * 2 * np.pi / num_agents_init for i in range(num_agents_init)]) + goal_theta
    goal_vertices = goal_pos + object_length * np.array([
        [np.cos(angle), np.sin(angle)] for angle in base_angles
    ])

    goal_polygon = plt.Polygon(goal_vertices, ec="C5", fc="C5", alpha=0.5)
    ax_main.add_patch(goal_polygon)

    agent_colors = [cm.rainbow(i/num_agents_init) for i in range(num_agents_init)]
    for i, vertex in enumerate(goal_vertices):
        vertex_circle = plt.Circle(vertex, 0.02, color=agent_colors[i], alpha=0.8, zorder=5)
        ax_main.add_patch(vertex_circle)

    goal_center = plt.Circle(goal_pos, 0.05, color="C5", alpha=0.8)
    ax_main.add_patch(goal_center)

    base_angles = np.array([i * 2 * np.pi / num_agents_init for i in range(num_agents)])
    triangle_vertices = object_length * np.array([
        [np.cos(angle), np.sin(angle)] for angle in base_angles
    ])
    mask = np.arange(num_agents) < num_agents_init
    masked_triangle_vertices = triangle_vertices[mask]
    triangle_patch = plt.Polygon(masked_triangle_vertices, ec="C3", fc="none")
    ax_main.add_patch(triangle_patch)

    agent_patches = [
        plt.Circle((0, 0), agent_radius, color=agent_colors[ii], zorder=5)
        for ii in range(num_agents_init)
    ]
    for patch in agent_patches:
        ax_main.add_patch(patch)
    
    # Initialize action arrows using quiver plots for better dynamic updates
    arrow_scale = 0.3  # Scale factor for action arrows
    action_quivers = []
    for ii in range(num_agents_init):
        # Create quiver plot for each agent's actions
        quiver = ax_main.quiver(0, 0, 0, 0, 
                               color=agent_colors[ii], 
                               alpha=0.8, 
                               scale=1/arrow_scale,  # Inverse scale for quiver
                               width=0.005,
                               zorder=6)
        action_quivers.append(quiver)

    text_font_opts = dict(size=16, color="k", family="sans-serif", weight="normal", transform=ax_main.transAxes)
    goal_text = ax_main.text(0.99, 1.00, "dist_goal=0", va="bottom", ha="right", **text_font_opts)
    obs_text = ax_main.text(0.99, 1.04, "dist_obs=0", va="bottom", ha="right", **text_font_opts)
    cost_text = ax_main.text(0.99, 1.12, "cost=0", va="bottom", ha="right", **text_font_opts)
    kk_text = ax_main.text(0.99, 1.08, "kk=0", va="bottom", ha="right", **text_font_opts)
    texts = [goal_text, obs_text, kk_text, cost_text]

    edge_col = LineCollection([], colors=[], linewidths=2, alpha=0.5, zorder=3)
    ax_main.add_collection(edge_col)
    
    # Initialize trajectory tracking
    trajectory_data = [[] for _ in range(num_agents_init)]  # Store x-positions for each agent
    trajectory_lines = []
    for i in range(num_agents_init):
        line, = ax_traj.plot([], [], color=agent_colors[i], linewidth=2, alpha=0.7, label=f'Agent {i}')
        trajectory_lines.append(line)
    ax_traj.legend(loc='upper right', fontsize=10)

    def init_fn():
        return [triangle_patch, *agent_patches, *texts, edge_col, *trajectory_lines, *action_quivers]

    def update(kk):
        if kk >= len(graph_list):
            return [triangle_patch, *agent_patches, *texts, edge_col, *trajectory_lines, *action_quivers]

        cur_graph = graph_list[kk]
        cur_cost = cost_list[kk] if kk < len(cost_list) else 0
        cur_reward = rewards_list[kk] if kk < len(rewards_list) else 0
        cur_env_state = cur_graph.env_states

        num_agents_cur = int(cur_env_state.real_num_agents)
        for ii in range(min(num_agents_cur, len(agent_patches))):
            pos = cur_env_state.a_pos[ii]
            agent_patches[ii].set_center(pos)
            
            # Track x-position for trajectory plot
            if ii < len(trajectory_data):
                trajectory_data[ii].append(pos[0])  # x-coordinate
            
            # Update action arrows using quiver plots
            if ii < len(action_quivers) and kk < len(action_list):
                action = action_list[kk]
                if len(action) > ii:
                    # Get action for this agent (assuming 2D actions: [x, y])
                    agent_action = action[ii]
                    if len(agent_action) >= 2:
                        # Set quiver position and direction
                        action_quivers[ii].set_offsets([pos[0], pos[1]])  # Start position
                        action_quivers[ii].set_UVC(agent_action[0], agent_action[1])  # Direction vector
                        
                        # Debug print for first few frames
                        if kk < 3 and ii == 0:
                            print(f"Frame {kk}, Agent {ii}: pos={pos}, action={agent_action[:2]}")
                    else:
                        # Hide arrow if no valid action
                        action_quivers[ii].set_offsets([pos[0], pos[1]])
                        action_quivers[ii].set_UVC(0, 0)
                else:
                    # Hide arrow if no action data
                    action_quivers[ii].set_offsets([pos[0], pos[1]])
                    action_quivers[ii].set_UVC(0, 0)
            else:
                # Hide arrow if no action data
                action_quivers[ii].set_offsets([pos[0], pos[1]])
                action_quivers[ii].set_UVC(0, 0)

        angle = float(np.array(cur_env_state.object_angle)[0, 0])
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_vertices = np.dot(masked_triangle_vertices, rotation_matrix.T)
        obj_center_coord = np.array(cur_env_state.object_pos)[0]
        transformed_vertices = rotated_vertices + obj_center_coord
        triangle_patch.set_xy(transformed_vertices)

        dist_goal = np.linalg.norm(cur_env_state.object_pos - cur_env_state.goal_pos)
        def _fmt_scalar(x):
            try:
                return f"{float(np.asarray(x)):.3f}"
            except Exception:
                try:
                    # Try taking first element if it's an array
                    return f"{float(np.asarray(x).ravel()[0]):.3f}"
                except Exception:
                    return str(x)
        cost_str = _fmt_scalar(cur_cost)
        reward_str = _fmt_scalar(cur_reward)
        obs_text.set_text("dist_obs=N/A")
        goal_text.set_text(f"dist_goal={dist_goal:.3f}")
        kk_text.set_text(f"kk={kk:04d}")
        cost_text.set_text(f"cost={cost_str}, reward={reward_str}")

        dim = 2
        all_pos = np.array(cur_graph.states[:total_nodes, :dim])
        edge_index = np.stack([np.array(cur_graph.senders), np.array(cur_graph.receivers)], axis=0)
        is_pad = np.any(edge_index == total_nodes, axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        if e_edge_index.shape[1] > 0:
            e_start = all_pos[e_edge_index[0, :]]
            e_end = all_pos[e_edge_index[1, :]]
            e_lines = np.stack([e_start, e_end], axis=1)
            e_is_goal = (e_edge_index[0, :] >= n_agent) & (e_edge_index[0, :] < n_agent + n_goal)
            e_colors = ["#2fdd00" if e_is_goal[ii] else "0.2" for ii in range(e_lines.shape[0])]
        else:
            e_lines = []
            e_colors = []

        edge_col.set_segments(e_lines)
        edge_col.set_color(e_colors)
        
        # Update trajectory plots
        for i in range(len(trajectory_lines)):
            if i < len(trajectory_data) and len(trajectory_data[i]) > 0:
                time_steps = list(range(len(trajectory_data[i])))
                trajectory_lines[i].set_data(time_steps, trajectory_data[i])
        
        # Update trajectory plot limits
        if kk > 0:
            ax_traj.set_xlim(0, max(kk + 1, 10))  # Show at least 10 time steps
            if any(trajectory_data):
                all_x_values = [x for traj in trajectory_data for x in traj]
                if all_x_values:
                    y_min = min(all_x_values) - 0.1
                    y_max = max(all_x_values) + 0.1
                    ax_traj.set_ylim(y_min, y_max)

        return [triangle_patch, *agent_patches, *texts, edge_col, *trajectory_lines, *action_quivers]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    anim_T = len(graph_list) if max_frames is None else min(len(graph_list), int(max_frames))
    ani = FuncAnimation(plt.gcf(), update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)

    # Extract timestamp from pickle filename
    pickle_stem = pathlib.Path(pickle_path).stem
    if "experiment_data_" in pickle_stem:
        timestamp = pickle_stem.split("experiment_data_")[-1]
    else:
        # Fallback to current time if no timestamp found
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output path with timestamp from pickle file
    base_out_path = pathlib.Path(pickle_path).parent / f"experiment_video_{timestamp}.mp4"
    out_path = base_out_path
    
    # Add suffix if file already exists
    counter = 1
    while out_path.exists():
        out_path = pathlib.Path(pickle_path).parent / f"experiment_video_{timestamp}_{counter}.mp4"
        counter += 1
    
    try:
        from model_dgppo.utils.utils import save_anim
        save_anim(ani, out_path)
        print(f"Video saved to: {out_path}")
    except Exception:
        gif_path = str(out_path).replace('.mp4', '.gif')
        ani.save(gif_path, writer='pillow', fps=fps)
        print(f"Video saved as GIF: {gif_path}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render experiment video from .pkl/.pth file")
    parser.add_argument("path", help="Path to experiment data (.pkl or .pth)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames rendered")
    args = parser.parse_args()
    make_exper_video_from_file(args.path, max_frames=args.max_frames)