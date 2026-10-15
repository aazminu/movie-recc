
import random
from typing import Dict, List

# In-memory storage for rooms 
# Structure:
# rooms = {
#   "ROOM123": {
#       "expected_members": 3,
#       "members": {
#           "user123": {"train_user_idx": 17, "is_done": True, "ratings": [...]}
#       },
#       "ratings": {},
#       "movie_list": [...],
#       "recommendation": {...}
#   }
# }
rooms: Dict[str, Dict] = {}

def create_or_join_room(room_code: str, user_id: str, expected_members: int, train_user_idx: int):
    """
    Creates a new room or you join an existing one
    """
    # IF room doesnt exist, create a new one 
    if room_code not in rooms:
        rooms[room_code] = {
            "expected_members": expected_members,
            "members": {},
            "ratings": {},
            "movie_list": [],
            "recommendation": None
        }

    # Room full check
    if len(rooms[room_code]["members"]) >= rooms[room_code]["expected_members"]:
        return None

    # Store member info including their training user index
    rooms[room_code]["members"][user_id] = {
        "train_user_idx": train_user_idx,
        "is_done": False
    }

    return rooms[room_code]

    # Stores user ratings for model
def store_ratings(room_code: str, user_id: str, ratings: List[Dict]):
    if room_code in rooms and user_id in rooms[room_code]["members"]:
        rooms[room_code]["members"][user_id]["ratings"] = ratings
        rooms[room_code]["members"][user_id]["is_done"] = True

    # checks if all users in roomk are done
def all_users_done(room_code: str):
    return all(member["is_done"] for member in rooms[room_code]["members"].values())
    
    # stores movie list that users will rate
def set_movie_list(room_code: str, movie_list: List[Dict]):
    rooms[room_code]["movie_list"] = movie_list

    # returns movie list for a room
def get_movie_list(room_code: str):
    return rooms[room_code]["movie_list"]

    #stores final group reccomendation 
def store_recommendation(room_code: str, rec: Dict):
    rooms[room_code]["recommendation"] = rec

    #returns final room recommendation
def get_recommendation(room_code: str):
    return rooms[room_code]["recommendation"]

    #returns list of user ids in the given room
def get_room_members(room_code: str):
    if room_code in rooms:
        return list(rooms[room_code]["members"].keys())
    return []


def get_all_ratings(room_code: str):
    """
    Return a list of all user ratings in the room.
    Each rating dict will include the user's assigned index
    """
    room = rooms.get(room_code)
    if not room:
        return []
    data = []
    for user_id, info in room["members"].items():
        for rating in info.get("ratings", []):
            # Include the user's training index in each rating
            rating["user_id"] = user_id
            rating["train_user_idx"] = info.get("train_user_idx", 0)
            data.append(rating)
    return data

# removes a member, not reallt used tbh 
def remove_member(room_code: str, user_id: str):
    if room_code in rooms and user_id in rooms[room_code]["members"]:
        del rooms[room_code]["members"][user_id]
