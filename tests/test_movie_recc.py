import numpy as np
from backend.rooms import (
    all_users_done,
    create_or_join_room,
    get_all_ratings,
    get_room_members,
    rooms,
    store_ratings,
)
from backend.model_service import recommend_for_group
import backend.main as main_app


def setup_function():
    rooms.clear()


def test_main_app_imports_cleanly_without_model_weights():
    assert main_app.app is not None


def test_room_join_capacity_and_completion():
    assert create_or_join_room("ROOM1", "alice", expected_members=2, train_user_idx=1) is not None
    # Single member finished should NOT trigger all_users_done when expected_members=2
    store_ratings("ROOM1", "alice", [{"movie_id": 1, "rating": 5.0}])
    assert all_users_done("ROOM1") is False

    # Re-joining with existing user_id preserves state
    assert create_or_join_room("ROOM1", "alice", expected_members=2, train_user_idx=1) is not None
    assert all_users_done("ROOM1") is False

    # Second member joins; capacity reached
    assert create_or_join_room("ROOM1", "bob", expected_members=2, train_user_idx=2) is not None
    assert create_or_join_room("ROOM1", "charlie", expected_members=2, train_user_idx=3) is None
    assert set(get_room_members("ROOM1")) == {"alice", "bob"}
    assert all_users_done("ROOM1") is False

    store_ratings("ROOM1", "bob", [{"movie_id": 2, "rating": 4.0}])
    assert all_users_done("ROOM1") is True
    all_r = get_all_ratings("ROOM1")
    assert len(all_r) == 2


def test_recommend_for_group_residual_bias_and_equal_member_weighting():
    catalog = [
        {"movie_id": 1, "title": "Rated Movie A", "genres": np.array([1.0, 0.0])},
        {"movie_id": 2, "title": "Rated Movie B", "genres": np.array([0.0, 1.0])},
        {"movie_id": 3, "title": "Sci-Fi Epic", "genres": np.array([1.0, 0.0])},
        {"movie_id": 4, "title": "Indie Drama", "genres": np.array([0.0, 1.0])},
    ]

    def predictor(train_user_idx: int, movie_id: int, genres: np.ndarray) -> float:
        return 3.0 if genres[0] > 0.5 else 2.5

    group_ratings = [
        {"user_id": "alice", "train_user_idx": 1, "movie_id": 1, "rating": 4.5},
        {"user_id": "alice", "train_user_idx": 1, "movie_id": 1, "rating": 4.5},
        {"user_id": "alice", "train_user_idx": 1, "movie_id": 1, "rating": 4.5},
        {"user_id": "bob", "train_user_idx": 2, "movie_id": 2, "rating": 2.0},
    ]

    recs = recommend_for_group(group_ratings, candidate_movies=catalog, predictor=predictor, top_k=3)
    assert len(recs) == 2
    assert recs[0]["title"] == "Sci-Fi Epic"
    assert recs[0]["score"] == 3.5
    assert recs[1]["title"] == "Indie Drama"
    assert recs[1]["score"] == 3.0


def test_recommend_for_group_with_real_movielens_catalog():
    group_ratings = [
        {"user_id": "u1", "train_user_idx": 0, "movie_id": 1, "rating": 5.0},
        {"user_id": "u2", "train_user_idx": 1, "movie_id": 2, "rating": 4.0},
    ]
    recs = recommend_for_group(group_ratings, top_k=3)
    assert len(recs) == 3
    assert all("title" in r and "score" in r for r in recs)
