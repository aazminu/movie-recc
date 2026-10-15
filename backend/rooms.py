from typing import Dict, List, Optional

rooms: Dict[str, Dict] = {}


def create_or_join_room(
    room_code: str,
    user_id: str,
    expected_members: int,
    train_user_idx: int,
) -> Optional[Dict]:
    """Create a new room or join an existing one."""
    if expected_members <= 0:
        raise ValueError("expected_members must be positive")

    if room_code not in rooms:
        rooms[room_code] = {
            "expected_members": expected_members,
            "members": {},
            "ratings": {},
            "movie_list": [],
            "recommendation": None,
        }

    room = rooms[room_code]
    if user_id not in room["members"] and len(room["members"]) >= room["expected_members"]:
        return None

    existing = room["members"].get(user_id)
    room["members"][user_id] = {
        "train_user_idx": train_user_idx,
        "is_done": existing["is_done"] if existing else False,
        "ratings": existing.get("ratings", []) if existing else [],
    }
    return room


def store_ratings(room_code: str, user_id: str, ratings: List[Dict]) -> None:
    if room_code in rooms and user_id in rooms[room_code]["members"]:
        rooms[room_code]["members"][user_id]["ratings"] = list(ratings)
        rooms[room_code]["members"][user_id]["is_done"] = True


def all_users_done(room_code: str) -> bool:
    room = rooms.get(room_code)
    if not room:
        return False
    members = room["members"]
    if len(members) < room["expected_members"] or not members:
        return False
    return all(member.get("is_done", False) for member in members.values())


def set_movie_list(room_code: str, movie_list: List[Dict]) -> None:
    if room_code in rooms:
        rooms[room_code]["movie_list"] = movie_list


def get_movie_list(room_code: str) -> List[Dict]:
    room = rooms.get(room_code)
    return room["movie_list"] if room else []


def store_recommendation(room_code: str, rec: List[Dict]) -> None:
    if room_code in rooms:
        rooms[room_code]["recommendation"] = rec


def get_recommendation(room_code: str):
    room = rooms.get(room_code)
    return room["recommendation"] if room else None


def get_room_members(room_code: str) -> List[str]:
    if room_code in rooms:
        return list(rooms[room_code]["members"].keys())
    return []


def get_all_ratings(room_code: str) -> List[Dict]:
    """Return a list of all user ratings in the room with user_id and train_user_idx."""
    room = rooms.get(room_code)
    if not room:
        return []
    data = []
    for user_id, info in room["members"].items():
        for rating in info.get("ratings", []):
            entry = dict(rating)
            entry["user_id"] = user_id
            entry["train_user_idx"] = info.get("train_user_idx", 0)
            data.append(entry)
    return data


def remove_member(room_code: str, user_id: str) -> None:
    if room_code in rooms and user_id in rooms[room_code]["members"]:
        del rooms[room_code]["members"][user_id]
