# backend/main.py
import joblib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
## ML MOdel 
from .model_service import recommend_for_group
#Connection manager
from .state import ConnectionManager
# Room related functions 
from .rooms import (
    create_or_join_room,
    store_ratings,
    all_users_done,
    set_movie_list,
    store_recommendation,
    get_room_members,
    get_all_ratings
)
#Init fast API app
app = FastAPI()

# User encoder to map usernames to models indexes
user_encoder = joblib.load("user_encoder.pkl")

# Connection manager which tracks websocket sessions
manager = ConnectionManager()

# Mount the frontend directory to serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def get():
    # Serve the index.html file directly
    return FileResponse("frontend/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Accept the connection immediately
    await websocket.accept()
    try:
        # First message will be a join_room message
        raw_data = await websocket.receive_text()
        data = json.loads(raw_data)
        if data.get("type") != "join_room":
            print("Expected join_room message, got:", data)
            await websocket.close(code=1003)
            return

        room_code = data["room_code"]
        user_id = data["user_id"]

        #Default to 3 people limit
        expected_members = data.get("expected_members", 3)

        # Encode user_id into training index in the model 
        try:
            train_user_idx = int(user_encoder.transform([user_id])[0])
        
        except ValueError:
            print(f"User {user_id} not found in encoder. Defaulting to 0.")
            train_user_idx = 0      
        

        # Add the connection to the specified room
        await manager.connect(websocket, room_code,user_id)
        room_data = create_or_join_room(room_code, user_id, expected_members, train_user_idx)
        
        
        if not room_data:
            # Room is full; notify the client and close connection.
            print(f"Room {room_code} is full. Rejecting user {user_id}.")
            await manager.send_personal_message({
                "type": "room_full",
                "message": "Room is already full. Cannot join."
            }, websocket)
            await websocket.close()
            return

        # Notify all users in the room of new user update 
        await manager.broadcast_to_room(room_code, {
            "type": "room_joined",
            "room_code": room_code,
            "members": get_room_members(room_code)
        })
        print(f"User {user_id} joined room {room_code}. Current members: {get_room_members(room_code)}", flush=True)

        #Main loop, handles incoming messages from the client 
        while True:

            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            msg_type = data.get("type")

            if msg_type == "request_movies":
                # Client is requesting a movie list 
                room_code = data["room_code"]
                genre = data.get("genre", "All")
                print(f" Received movie request for genre: {genre} in room: {room_code}")
                
                # Get the list and print out the confirmation
                from .data_service import get_enriched_movie_list
                movie_list = get_enriched_movie_list(sample_size=5, genre=genre)
                print(f"Sampled {len(movie_list)} movies for room {room_code}")
                
                # Store movie list and then send it to the client 
                set_movie_list(room_code, movie_list)
                await manager.send_personal_message({
                    "type": "movie_list",
                    "room_code": room_code,
                    "movies": movie_list
                }, websocket)

            # User submits ratings 
            elif msg_type == "rating_submission":
                room_code = data["room_code"]
                user_id = data["user_id"]
                ratings = data["ratings"]

                store_ratings(room_code, user_id, ratings)
                # Check for all users 
                if all_users_done(room_code):
                    print(f"All users in room {room_code} submitted ratings.")
                    await manager.broadcast_to_room(room_code, {
                        "type": "all_submitted",
                        "room_code": room_code,
                        "message": "All ratings submitted. Click 'Show Group Recommendation' to view the recommendation."
                    })

            # Trigger the group reccomendation
            elif msg_type == "trigger_group_rec":
                all_ratings = get_all_ratings(room_code)
                if not all_ratings:
                    print(f"No ratings found for room {room_code}. Cannot generate recommendation.")
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "No ratings available to generate a recommendation."
                    }, websocket)
                    continue  

                # Compute group recommendation using the ml model
                final_recommendation = recommend_for_group(all_ratings)
                store_recommendation(room_code, final_recommendation)
                print("Final group recommendation:", final_recommendation, flush=True)

                # Broadcast the recommendation to everyone in the room
                await manager.broadcast_to_room(room_code, {
                    "type": "group_recommendation",
                    "room_code": room_code,
                    "movies": final_recommendation
                })
        else: 
                print("Un"
                "known message type received:", msg_type) 
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("A user disconnected from room.")
