import numpy as np

train_data = np.load('./dppo/log/robomimic/can/train.npz')
states = train_data['states']  # Shape: (N, 23)
actions = train_data['actions']  # Shape: (N, 7)
traj_lengths = train_data['traj_lengths'] # Shape: (N, )

print("states shape: ", states.shape)
print("actions shape: ", actions.shape)
print("traj lengths shape: ", traj_lengths.shape)

better_1_start = 0
better_1_end = np.sum(traj_lengths[0:50])

worse_1_start = better_1_end
worse_1_end = worse_1_start + np.sum(traj_lengths[50:100])

okay_1_start = worse_1_end
okay_1_end = okay_1_start + np.sum(traj_lengths[100:150])

better_2_start = okay_1_end
better_2_end = better_2_start + np.sum(traj_lengths[150:200])

worse_2_start = better_2_end
worse_2_end = worse_2_start + np.sum(traj_lengths[200:250])

okay_2_start = worse_2_end
okay_2_end = okay_2_start + np.sum(traj_lengths[250:300])

worse_states = np.append(states[worse_1_start:worse_1_end, :], states[worse_2_start:worse_2_end, :], axis=0)
worse_actions = np.append(actions[worse_1_start:worse_1_end, :], actions[worse_2_start:worse_2_end, :], axis=0)
worse_traj_lengths = np.append(traj_lengths[50:100], traj_lengths[200:250], axis=0)
print("worse states shape: ", worse_states.shape)
print("worse actions shape: ", worse_actions.shape)
print("worse traj lengths shape: ", worse_traj_lengths.shape)
print(f"Min: {np.min(worse_traj_lengths):.6f}")
print(f"Max: {np.max(worse_traj_lengths):.6f}")
print("\n\n")

okay_states = np.append(states[okay_1_start:okay_1_end, :], states[okay_2_start:okay_2_end, :], axis=0)
okay_actions = np.append(actions[okay_1_start:okay_1_end, :], actions[okay_2_start:okay_2_end, :], axis=0)
okay_traj_lengths = np.append(traj_lengths[100:150], traj_lengths[250:300], axis=0)
print("okay states shape: ", okay_states.shape)
print("okay actions shape: ", okay_actions.shape)
print("okay traj lengths shape: ", okay_traj_lengths.shape)
print(f"Min: {np.min(okay_traj_lengths):.6f}")
print(f"Max: {np.max(okay_traj_lengths):.6f}")
print("\n\n")

better_states = np.append(states[better_1_start:better_1_end, :], states[better_2_start:better_2_end, :], axis=0)
better_actions = np.append(actions[better_1_start:better_1_end, :], actions[better_2_start:better_2_end, :], axis=0)
better_traj_lengths = np.append(traj_lengths[0:50], traj_lengths[150:200], axis=0)
print("better states shape: ", better_states.shape)
print("better actions shape: ", better_actions.shape)
print("better traj lengths shape: ", better_traj_lengths.shape)
print(f"Min: {np.min(better_traj_lengths):.6f}")
print(f"Max: {np.max(better_traj_lengths):.6f}")
print("\n\n")

np.savez_compressed(
    './dppo/log/robomimic/can/worse_train.npz', 
    states=worse_states, 
    actions=worse_actions,
    traj_lengths=worse_traj_lengths
)

np.savez_compressed(
    './dppo/log/robomimic/can/okay_train.npz', 
    states=okay_states, 
    actions=okay_actions,
    traj_lengths=okay_traj_lengths
)
np.savez_compressed(
    './dppo/log/robomimic/can/better_train.npz', 
    states=better_states, 
    actions=better_actions,
    traj_lengths=better_traj_lengths
)