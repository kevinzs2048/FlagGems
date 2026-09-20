import pytest
import torch
from qwen_image_cpu.accelerate_threads import ops, policy


def states():
    return {row[0]: row[4] for row in ops().pool_mode(-1).tolist()}


@pytest.mark.parametrize("mode", [0, 1])
def test_all_workers_set_and_restore(mode):
    torch.set_num_threads(4)
    before = states()
    assert len(before) == 4
    with policy(mode):
        assert set(states().values()) == {mode}
    assert states() == before


def test_exception_restores_and_invalid_mode_does_not_mutate():
    torch.set_num_threads(4)
    before = states()
    with pytest.raises(ValueError, match="injected"):
        with policy(1):
            raise ValueError("injected")
    assert states() == before
    with pytest.raises(RuntimeError):
        ops().pool_mode(2)
    assert states() == before
