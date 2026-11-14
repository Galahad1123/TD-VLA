from td_vla.annotation.stages import classify, stabilize


def state(**kw):
    s = dict(
        success=False,
        held=[False, False],
        lifted=False,
        near_target=False,
        object_in_basket=False,
        basket_held=False,
        first_placed=False,
        second_held=False,
        ambiguous=False,
    )
    s.update(kw)
    return s


def test_handover_distinguishes_dual_contact_and_receiving_arm():
    assert classify("handover_block", state(held=[True, False])) == (1, True)
    assert classify("handover_block", state(held=[True, True])) == (2, True)
    assert classify("handover_block", state(held=[False, True])) == (3, True)
    assert classify("handover_block", state(lifted=True)) == (0, False)


def test_terminal_success_uses_official_flag():
    assert classify("move_stapler_pad", state(near_target=True, held=[True, False])) == (2, True)
    assert classify("move_stapler_pad", state(success=True)) == (3, True)
    assert classify("lift_pot", state(held=[True, False]))[1] is False


def test_recovery_not_forced_monotonic_and_boundary_invalid():
    labels, valid = stabilize([(0, True), (1, True), (1, True), (0, True), (0, True), (3, True)], 3)
    assert labels == [0, 1, 1, 0, 0, 3]
    assert valid == [True, False, True, False, True, True]


def test_unreliable_state_not_hidden_by_smoothing():
    _, valid = stabilize([(1, True), (1, False), (1, True), (1, True)], 3)
    assert valid == [True, False, False, True]


def test_secondary_block_contact_loss_and_airborne_first_block_are_invalid():
    import numpy as np
    from td_vla.annotation.stages import scene_features

    initial = np.array([[0.1, 0.1, 0.766], [0.2, 0.1, 0.766]])
    xyz = np.array([[0.0, -0.13, 0.766], [0.1, -0.1, 0.86]])
    contacts = np.zeros((2, 2, 2), dtype=bool)
    pair = np.zeros((2, 2), dtype=bool)

    def evaluate(points, touching):
        s = scene_features(
            "stack_blocks_two",
            points,
            initial,
            touching,
            np.zeros(14),
            pair,
            False,
            previous=points.copy(),
            dt=0.06,
        )
        return s, classify("stack_blocks_two", s)

    contacts[1, 0, 0] = True
    assert evaluate(xyz, contacts)[1][1] is False
    contacts[:] = False
    assert evaluate(xyz, contacts)[1][1] is False
    contacts[1, 0, :] = True
    assert evaluate(xyz, contacts)[1] == (3, True)
    contacts[:] = False
    xyz[1] = initial[1]
    xyz[0, 2] = 0.86
    s, result = evaluate(xyz, contacts)
    assert not s["first_placed"] and result[1] is False
    xyz[0, 2] = 0.766
    assert evaluate(xyz, contacts)[1] == (2, True)


def test_basket_single_finger_support_is_not_valid():
    import numpy as np
    from td_vla.annotation.stages import scene_features

    initial = np.array([[0, 0, 0.75], [0, 0, 0.75]])
    xyz = initial + [0, 0, 0.06]
    contacts = np.zeros((2, 2, 2), dtype=bool)
    contacts[1, 1, 0] = True
    s = scene_features(
        "place_object_basket",
        xyz,
        initial,
        contacts,
        np.zeros(14),
        np.array([[False, True], [True, False]]),
        False,
        previous=xyz,
        dt=0.06,
    )
    assert classify("place_object_basket", s)[1] is False


def test_contradictory_airborne_placement_is_rejected():
    assert classify("stack_blocks_two", state(lifted=True, first_placed=True))[1] is False


def test_touching_basket_while_holding_object_remains_object_placement():
    s = state(held=[True, False], object_in_basket=True, basket_held=True)
    assert classify("place_object_basket", s) == (1, True)
