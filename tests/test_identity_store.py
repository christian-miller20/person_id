import numpy as np

from person_id_pi.identity_store import IdentityStore


def test_rename_user_success_preserves_templates(tmp_path):
    store = IdentityStore(tmp_path / "store.json")
    store.add_template("alice", np.asarray([1.0, 0.0, 0.0], dtype=np.float32))

    renamed = store.rename_user("alice", "alicia")

    assert renamed is True
    assert store.has_user("alice") is False
    assert store.has_user("alicia") is True
    templates = store.get_templates("alicia")
    assert len(templates) == 1
    assert np.allclose(templates[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_rename_user_rejects_existing_target(tmp_path):
    store = IdentityStore(tmp_path / "store.json")
    store.add_template("alice", np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    store.add_template("bob", np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

    renamed = store.rename_user("alice", "bob")

    assert renamed is False
    assert store.has_user("alice") is True
    assert store.has_user("bob") is True


def test_rename_user_rejects_missing_source(tmp_path):
    store = IdentityStore(tmp_path / "store.json")

    renamed = store.rename_user("missing", "new_id")

    assert renamed is False
    assert store.has_user("new_id") is False


def test_rename_user_same_name_is_no_op_success(tmp_path):
    store = IdentityStore(tmp_path / "store.json")
    store.add_user("alice")

    renamed = store.rename_user("alice", "alice")

    assert renamed is True
    assert store.has_user("alice") is True
