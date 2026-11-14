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
