
# Internal
import os
import sys

# External
import numpy as np
import pytest

# Path Append
sys.path.append(os.path.abspath(os.curdir))

# Internal
from memory.replaybuffer import ReplayBuffer

def test_initialization():
    obs_shape = (3, 84, 84)
    size = 1000
    batch_size = 32
    buffer = ReplayBuffer(obs_shape, size, batch_size)

    assert buffer.obs_buf.shape == (size, *obs_shape)
    assert buffer.next_obs_buf.shape == (size, *obs_shape)
    assert buffer.acts_buf.shape == (size,)
    assert buffer.rews_buf.dtype == np.float32
    assert buffer.done_buf.dtype == np.bool_

def test_store_and_size():
    obs_shape = (3, 84, 84)
    size = 100
    batch_size = 32
    buffer = ReplayBuffer(obs_shape, size, batch_size)
    obs = np.random.randint(0, 256, size=obs_shape, dtype=np.uint8)
    act = 1
    reward = 1.0
    next_obs = np.random.randint(0, 256, size=obs_shape, dtype=np.uint8)
    done = False

    for _ in range(50):
        buffer.store(obs, act, reward, next_obs, done)

    assert len(buffer) == 50
    assert buffer.size == 50
    buffer.store(obs, act, reward, next_obs, done)
    assert len(buffer) == 51

def test_sampling():
    obs_shape = (3, 84, 84)
    size = 100
    batch_size = 10
    buffer = ReplayBuffer(obs_shape, size, batch_size)

    # Store enough samples for at least two batches
    for _ in range(20):
        obs = np.random.randint(0, 256, size=obs_shape, dtype=np.uint8)
        act = np.random.randint(0, 10)
        reward = np.random.random()
        next_obs = np.random.randint(0, 256, size=obs_shape, dtype=np.uint8)
        done = np.random.choice([True, False])
        buffer.store(obs, act, reward, next_obs, done)

    batch = buffer.sample_batch()
    assert batch['obs'].shape == (batch_size, *obs_shape)
    assert batch['next_obs'].shape == (batch_size, *obs_shape)
    assert batch['acts'].shape == (batch_size,)
    assert batch['rews'].shape == (batch_size,)
    assert batch['done'].shape == (batch_size,)

    # Testing attempt to sample more than available without replacement
    with pytest.raises(ValueError):
        small_buffer = ReplayBuffer(obs_shape, size, batch_size)
        small_buffer.sample_batch()

if __name__ == '__main__':
    pytest.main()
