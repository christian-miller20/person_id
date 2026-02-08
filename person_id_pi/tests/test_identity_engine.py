import numpy as np

from person_id_pi.face_types import FaceEmbedding
from person_id_pi.identity_config import IdentityConfig
from person_id_pi.identity_engine import IdentityEngine
from person_id_pi.identity_store import IdentityStore


def _embed(vec, quality=1.0):
    return FaceEmbedding(
        embedding=np.asarray(vec, dtype=np.float32),
        quality=quality,
        bbox=(0, 0, 10, 10),
        landmarks=None,
    )


def test_aggregate_tracklet_filters_by_quality(tmp_path):
    store = IdentityStore(tmp_path / "store.json")
    config = IdentityConfig(quality_min=0.6)
    engine = IdentityEngine(store=store, config=config)
    good = _embed([1.0, 0.0, 0.0], quality=0.9)
    bad = _embed([0.0, 1.0, 0.0], quality=0.2)
    tracklet = engine.aggregate_tracklet([good, bad])
    assert tracklet.n_used == 1


def test_match_accepts_best_user(tmp_path):
    store = IdentityStore(tmp_path / "store.json")
    store.add_user("alice")
    store.add_template("alice", np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    store.add_user("bob")
    store.add_template("bob", np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

    config = IdentityConfig(accept_threshold=0.5, margin_threshold=0.1, n_min=1)
    engine = IdentityEngine(store=store, config=config)
    tracklet = engine.aggregate_tracklet([_embed([1.0, 0.0, 0.0])])
    decision = engine.match(tracklet)
    assert decision.accepted is True
    assert decision.user_id == "alice"


def test_match_rejects_low_margin(tmp_path):
    store = IdentityStore(tmp_path / "store.json")
    store.add_user("alice")
    store.add_template("alice", np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    store.add_user("bob")
    store.add_template("bob", np.asarray([0.9, 0.1, 0.0], dtype=np.float32))

    config = IdentityConfig(accept_threshold=0.5, margin_threshold=0.2, n_min=1)
    engine = IdentityEngine(store=store, config=config)
    tracklet = engine.aggregate_tracklet([_embed([1.0, 0.0, 0.0])])
    decision = engine.match(tracklet)
    assert decision.accepted is False
    assert decision.user_id is None
