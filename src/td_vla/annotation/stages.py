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
    valid = not state["ambiguous"] and not state.get("unsupported", False)
    if task == "stack_blocks_two" and state["first_placed"] and state["lifted"]:
        valid = False
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
        # Finger contact with the basket during insertion is not yet basket lifting.
        label = (
            1
            if held
            else 3
            if state["basket_held"] and state["object_in_basket"]
            else 2
            if state["object_in_basket"]
            else 0
        )
        return label, valid and (held or state["object_in_basket"] or not state["lifted"])
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


def scene_features(
    task, xyz, initial, contacts, qpos, object_contact, success, previous=None, dt=0.06
):
    """Causal evidence for all manipulated objects, including secondary-object support."""
    import numpy as np

    xyz, initial, contacts = np.asarray(xyz), np.asarray(initial), np.asarray(contacts)
    held = contacts.all(-1) & (np.asarray(qpos)[None, [6, 13]] < 0.8)
    lifted = xyz[:, 2] - initial[:, 2] > 0.025
    relevant = [0, 1] if task in ("stack_blocks_two", "place_object_basket") else [0]
    ambiguous = bool(np.any(contacts[relevant].sum(-1) == 1))
    s = dict(
        success=bool(success),
        held=held[0].tolist(),
        lifted=bool(lifted[0]),
        near_target=bool(np.linalg.norm(xyz[0, :2] - xyz[1, :2]) < 0.04) if len(xyz) > 1 else False,
        object_in_basket=False,
        basket_held=False,
        first_placed=False,
        second_held=False,
        ambiguous=ambiguous,
        unsupported=bool(lifted[0] and not held[0].any()),
    )
    if task == "stack_blocks_two":
        speed = (
            np.linalg.norm(xyz[0] - previous[0]) / dt
            if previous is not None and dt > 0
            else float("inf")
        )
        first_supported = bool(abs(xyz[0, 2] - initial[0, 2]) < 0.02 and speed < 0.05)
        second_supported = bool(object_contact[0, 1] and abs(xyz[1, 2] - xyz[0, 2] - 0.05) < 0.015)
        s.update(
            first_placed=bool(
                np.linalg.norm(xyz[0, :2] - [0, -0.13]) < 0.035
                and not held[0].any()
                and first_supported
            ),
            second_held=bool(held[1].any()),
            first_supported=first_supported,
            second_supported=second_supported,
            unsupported=bool(
                (lifted[0] and not held[0].any() and not first_supported)
                or (lifted[1] and not held[1].any() and not second_supported)
            ),
        )
    elif task == "place_object_basket":
        inside = bool(object_contact[0, 1] and np.linalg.norm(xyz[0] - xyz[1]) < 0.15)
        s.update(
            object_in_basket=inside,
            basket_held=bool(held[1].any()),
            unsupported=bool(
                (lifted[0] and not held[0].any() and not inside)
                or (lifted[1] and not held[1].any())
            ),
        )
    return s
