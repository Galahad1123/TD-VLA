import pytest
from types import SimpleNamespace
from td_vla.integrations import expert_failures


def test_no_grasp_pose_is_a_failed_expert_attempt():
    action = {}
    exec(
        compile(
            'def make_action(target_pose):\n    assert target_pose is not None, "target_pose cannot be None for move action."\n',
            "/robotwin/envs/utils/action.py",
            "exec",
        ),
        action,
    )
    grasp = {"make_action": action["make_action"]}
    exec(
        compile(
            "def grasp_actor():\n    pre_grasp_pose = grasp_pose = None\n    make_action(pre_grasp_pose)\n",
            "/robotwin/envs/_base_task.py",
            "exec",
        ),
        grasp,
    )
    result = expert_failures.play_expert(SimpleNamespace(play_once=grasp["grasp_actor"]))
    assert result["reason"] == "no_grasp_pose"
    assert "target_pose cannot be None" in result["message"]


def test_unrelated_assertions_and_system_errors_still_propagate():
    for error in [
        AssertionError("target_pose cannot be None for move action."),
        RuntimeError("CUDA out of memory"),
    ]:

        def fail():
            raise error

        with pytest.raises(type(error)):
            expert_failures.play_expert(SimpleNamespace(play_once=fail))


def test_successful_expert_execution_is_unchanged():
    assert expert_failures.play_expert(SimpleNamespace(play_once=lambda: None)) is None
