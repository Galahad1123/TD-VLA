"""Task-local, causal scene-state rules. IDs are not comparable across tasks."""

VOCAB = {
    "move_stapler_pad": ["approach_grasp", "held_transport", "align_place", "complete"],
    "lift_pot": ["approach_grasp", "bilateral_hold_lift", "complete"],
    "handover_block": [
        "approach_grasp",
        "giver_holds",
        "dual_hold",
        "receiver_holds_place",
        "complete",
    ],
    "beat_block_hammer": ["approach_grasp", "held_transport", "align_strike", "complete"],
    "stack_blocks_two": [
        "first_approach",
        "first_held",
        "first_placed_second_approach",
        "second_held_stack",
        "complete",
    ],
    "place_object_basket": [
        "approach_object",
        "object_held",
        "object_in_basket",
        "basket_held_lift",
        "complete",
    ],
}


def classify(task, state):
    if state["success"]:
        return len(VOCAB[task]) - 1, True
    left, right = state["held"]
    held = left or right
    valid = not state["ambiguous"]
    if task in ("move_stapler_pad", "beat_block_hammer"):
        return (
            2 if state["near_target"] and held else 1 if held else 0,
            valid and (held or not state["lifted"]),
        )
    if task == "lift_pot":
        return (
            1 if left and right else 0,
            valid and (left == right) and (held or not state["lifted"]),
        )
    if task == "handover_block":
        return (
            2 if left and right else 3 if right else 1 if left else 0,
            valid and (held or not state["lifted"]),
        )
    if task == "stack_blocks_two":
        return (
            3 if state["second_held"] else 2 if state["first_placed"] else 1 if held else 0,
            valid
            and (held or state["first_placed"] or state["second_held"] or not state["lifted"]),
        )
    if task == "place_object_basket":
        return (
            3
            if state["basket_held"] and state["object_in_basket"]
            else 2
            if state["object_in_basket"]
            else 1
            if held
            else 0,
            valid and (held or state["object_in_basket"] or not state["lifted"]),
        )
    raise ValueError(task)


def stabilize(raw, terminal):
    labels, valid = [], []
    previous = None
    for label, ok in raw:
        stable = previous is None or previous == (label, True) or label == terminal
        labels.append(label)
        valid.append(bool(ok and stable))
        previous = (label, ok)
    return labels, valid
