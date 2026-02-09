import numpy as np

# Load normalization parameters
norm_data = np.load('./dppo/log/robomimic/can/normalization.npz')
obs_min = norm_data['obs_min']
obs_max = norm_data['obs_max']
action_min = norm_data['action_min']
action_max = norm_data['action_max']

# Load your training data
train_data = np.load('./dppo/log/robomimic/can/worse_train.npz')
raw_states = train_data['states']  # Shape: (N, 23)
raw_actions = train_data['actions']  # Shape: (N, 7)

# Normalize observations/states to [-1, 1]
normalized_states = 2 * (raw_states - obs_min) / (obs_max - obs_min + 1e-6) - 1

# Normalize actions to [-1, 1]
normalized_actions = 2 * (raw_actions - action_min) / (action_max - action_min + 1e-6) - 1

# Optional: Save normalized data
np.savez_compressed(
    './dppo/log/robomimic/can/worse_train.npz', 
    states=normalized_states, 
    actions=normalized_actions,
    rewards=train_data['rewards'],
    terminals=train_data['terminals'],
    traj_lengths=train_data['traj_lengths']
)

print(f"States shape: {normalized_states.shape}")
print(f"Actions shape: {normalized_actions.shape}")
print(f"States range: [{normalized_states.min():.3f}, {normalized_states.max():.3f}]")
print(f"Actions range: [{normalized_actions.min():.3f}, {normalized_actions.max():.3f}]")