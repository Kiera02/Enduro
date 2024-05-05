# **TODO**

- [x] Setup pre-commit

- [ ] Create Enduro env

- [ ] Preprocess:
    - [ ] Repeat Action in Frames
        - [ ] Step: Repeat 4 steps but just take 2 lastest observations and tke the maximum
    - [ ] Process Observation game env:
        - [ ] Transform observation: resize shape, norm 0 -> 1 / 255.0
- [ ] Replay Buffer: store and manage experience replay.
        - [ ] Save: state, action, reward, next state, and done.
        - [ ] Sample: sample a batch of experiences from memory. Return as individual np.array

- [ ] Deep Q-Network:
    - [ ] Declare network: cnn and self attention [Multihead Attention pytorch]
    - [ ] Forward, Backward
    - [ ] Convert input data into tensor and load it to the current device [GPU or CPU]

- [ ] DQN Agent:
    - [ ] Called Network: dqn and target dqn
    - [ ] Choose Action: compare with epsilon
    - [ ] Train agent
    - [ ] Replace target dqn
    - [ ] Decrement_epsilon: force model make decision

- [ ] Main:
