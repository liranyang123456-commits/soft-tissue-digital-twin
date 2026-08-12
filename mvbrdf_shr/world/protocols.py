"""Frozen scene-level evaluation protocols.

DEV4 was used for initial method design. PILOT6 was subsequently inspected
for protocol calibration and is therefore excluded from the final independent
TEST32 set. These lists are explicit so filesystem ordering cannot change a
reported split.
"""
from __future__ import annotations

STANFORD_ORB_ALL42: tuple[str, ...] = (
    "baking_scene001",
    "baking_scene002",
    "baking_scene003",
    "ball_scene002",
    "ball_scene003",
    "ball_scene004",
    "blocks_scene002",
    "blocks_scene005",
    "blocks_scene006",
    "cactus_scene001",
    "cactus_scene005",
    "cactus_scene007",
    "car_scene002",
    "car_scene004",
    "car_scene006",
    "chips_scene002",
    "chips_scene003",
    "chips_scene004",
    "cup_scene003",
    "cup_scene006",
    "cup_scene007",
    "curry_scene001",
    "curry_scene005",
    "curry_scene007",
    "gnome_scene003",
    "gnome_scene005",
    "gnome_scene007",
    "grogu_scene001",
    "grogu_scene002",
    "grogu_scene003",
    "pepsi_scene002",
    "pepsi_scene003",
    "pepsi_scene004",
    "pitcher_scene001",
    "pitcher_scene005",
    "pitcher_scene007",
    "salt_scene004",
    "salt_scene005",
    "salt_scene007",
    "teapot_scene001",
    "teapot_scene002",
    "teapot_scene006",
)

STANFORD_ORB_DEV4: tuple[str, ...] = (
    "ball_scene003",
    "blocks_scene005",
    "gnome_scene003",
    "teapot_scene002",
)

_dev = set(STANFORD_ORB_DEV4)
STANFORD_ORB_TEST38: tuple[str, ...] = tuple(
    scene for scene in STANFORD_ORB_ALL42 if scene not in _dev
)

STANFORD_ORB_PILOT6: tuple[str, ...] = (
    "baking_scene001",
    "ball_scene004",
    "cactus_scene005",
    "car_scene006",
    "grogu_scene002",
    "salt_scene007",
)

STANFORD_ORB_CALIB10: tuple[str, ...] = (
    *STANFORD_ORB_DEV4,
    *STANFORD_ORB_PILOT6,
)
_calibration = set(STANFORD_ORB_CALIB10)
STANFORD_ORB_TEST32: tuple[str, ...] = tuple(
    scene for scene in STANFORD_ORB_ALL42 if scene not in _calibration
)


def stanford_scene_group(name: str) -> tuple[str, ...]:
    """Return an immutable, predefined Stanford-ORB capture group."""

    groups = {
        "dev4": STANFORD_ORB_DEV4,
        "test38": STANFORD_ORB_TEST38,
        "all42": STANFORD_ORB_ALL42,
        "pilot6": STANFORD_ORB_PILOT6,
        "calib10": STANFORD_ORB_CALIB10,
        "test32": STANFORD_ORB_TEST32,
    }
    try:
        return groups[name]
    except KeyError as exc:
        raise ValueError(f"unknown Stanford-ORB scene group: {name}") from exc
