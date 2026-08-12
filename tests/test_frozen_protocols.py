from mvbrdf_shr.world.protocols import (
    STANFORD_ORB_ALL42,
    STANFORD_ORB_CALIB10,
    STANFORD_ORB_DEV4,
    STANFORD_ORB_PILOT6,
    STANFORD_ORB_TEST32,
    STANFORD_ORB_TEST38,
    stanford_scene_group,
)


def test_stanford_protocol_partitions_are_disjoint_and_complete():
    assert len(STANFORD_ORB_ALL42) == 42
    assert len(set(STANFORD_ORB_ALL42)) == 42
    assert len(STANFORD_ORB_DEV4) == 4
    assert len(STANFORD_ORB_TEST38) == 38
    assert set(STANFORD_ORB_DEV4).isdisjoint(STANFORD_ORB_TEST38)
    assert set(STANFORD_ORB_DEV4) | set(STANFORD_ORB_TEST38) == set(
        STANFORD_ORB_ALL42
    )


def test_inspected_pilot_is_excluded_from_final_independent_test():
    assert len(STANFORD_ORB_PILOT6) == 6
    assert set(STANFORD_ORB_PILOT6) <= set(STANFORD_ORB_TEST38)
    assert len(STANFORD_ORB_CALIB10) == 10
    assert len(STANFORD_ORB_TEST32) == 32
    assert set(STANFORD_ORB_CALIB10).isdisjoint(STANFORD_ORB_TEST32)
    assert set(STANFORD_ORB_CALIB10) | set(STANFORD_ORB_TEST32) == set(
        STANFORD_ORB_ALL42
    )


def test_named_groups_return_frozen_tuples():
    assert stanford_scene_group("dev4") == STANFORD_ORB_DEV4
    assert stanford_scene_group("test38") == STANFORD_ORB_TEST38
    assert stanford_scene_group("all42") == STANFORD_ORB_ALL42
    assert stanford_scene_group("pilot6") == STANFORD_ORB_PILOT6
    assert stanford_scene_group("calib10") == STANFORD_ORB_CALIB10
    assert stanford_scene_group("test32") == STANFORD_ORB_TEST32
