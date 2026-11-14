"""Recognize the pinned RoboTwin no-candidate-grasp failure without hiding bugs."""


def play_expert(task):
    try:
        task.play_once()
    except AssertionError as error:
        if str(error) != "target_pose cannot be None for move action.":
            raise
        frames = []
        tb = error.__traceback__
        while tb is not None:
            frames.append(tb.tb_frame)
            tb = tb.tb_next
        grasp = any(
            frame.f_code.co_filename.endswith("/envs/_base_task.py")
            and frame.f_code.co_name == "grasp_actor"
            and "pre_grasp_pose" in frame.f_locals
            and "grasp_pose" in frame.f_locals
            and frame.f_locals["pre_grasp_pose"] is None
            and frame.f_locals["grasp_pose"] is None
            for frame in frames
        )
        action = frames[-1].f_code.co_filename.endswith("/envs/utils/action.py")
        if not (grasp and action):
            raise
        return dict(reason="no_grasp_pose", exception=type(error).__name__, message=str(error))
    return None
