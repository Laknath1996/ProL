import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_Qvalues(Q_table, env, time_steps_to_plot=None, path=None):
    """
    Plot Q-values heatmap for different time steps and actions
    
    Args:
        Q_table: Shape (state_size, time_size, action_size)
        env: Environment instance
        time_steps_to_plot: List of time steps to visualize (if None, plot all)
    """
    grid_size = env.grid_size
    action_size = env.action_size
    
    action_names = ['Stay', 'Up', 'Right', 'Down', 'Left']
    
    # Create subplot grid
    fig, axes = plt.subplots(len(time_steps_to_plot), action_size, 
                            figsize=(4*action_size, 4*len(time_steps_to_plot)))
    
    if len(time_steps_to_plot) == 1:
        axes = axes[np.newaxis, :]
    
    for t_idx, t in enumerate(time_steps_to_plot):
        for a in range(action_size):
            # Reshape Q-values for this time step and action into 2D grid
            q_grid = Q_table[:, t, a].reshape(grid_size, grid_size)
            
            # Plot heatmap
            sns.heatmap(q_grid, 
                       ax=axes[t_idx, a],
                       cmap='RdYlBu',
                       annot=True,
                       fmt='.2f',
                       cbar=False)
            
            # Set titles
            if t_idx == 0:
                axes[t_idx, a].set_title(f'Action: {action_names[a]}')
            if a == 0:
                axes[t_idx, a].set_ylabel(f'Time step {t}')
    
    plt.tight_layout()
    plt.savefig(path + '/q_values.png')

