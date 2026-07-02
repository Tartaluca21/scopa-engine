"""Tests for the pygame-free menu selection state (gui.menu.MenuState)."""

from __future__ import annotations

from gui.menu import MODES, MenuState


def test_default_menu_is_single_deal_no_recording() -> None:
    state = MenuState()
    assert state.target is None
    assert state.record_moves is False
    assert state.to_config().target is None


def test_set_mode_selects_match_targets() -> None:
    state = MenuState()
    state.set_mode(1)
    assert state.target == 11.0
    state.set_mode(2)
    assert state.target == 21.0
    state.set_mode(0)
    assert state.target is None


def test_set_mode_wraps_out_of_range() -> None:
    state = MenuState()
    state.set_mode(len(MODES))  # wraps back to 0
    assert state.mode_index == 0


def test_toggle_record_flips_flag() -> None:
    state = MenuState()
    state.toggle_record()
    assert state.record_moves is True
    assert state.to_config().record_moves is True


def test_apply_actions() -> None:
    state = MenuState()
    assert state.apply("mode:1") is False
    assert state.target == 11.0
    assert state.apply("record") is False
    assert state.record_moves is True
    assert state.apply("start") is True  # start is the only True-returning action


def test_config_names_the_gui_bot() -> None:
    cfg = MenuState().to_config()
    assert cfg.bot_name == "PIMC(n_worlds=12,max_depth=5)"
