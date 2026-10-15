from fastapi import WebSocket
from typing import Dict, List
from .rooms import remove_member  
class ConnectionManager:
    """
    Mansges websocket connections, tracks users in rooms and handles broadcasting of messages to rooms or users
    """
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}  # room_code → list of websockets
        self.connection_info: Dict[WebSocket, tuple] = {}  # Map websocket to (room_code, user_id)

    # gets a new websocket connection under the room and given user
    async def connect(self, websocket: WebSocket, room_code: str, user_id: str):
        
        if room_code not in self.active_connections:
            self.active_connections[room_code] = []
        
        self.active_connections[room_code].append(websocket)
        self.connection_info[websocket] = (room_code, user_id)
    
    #removes a websocket and its associated info 
    def disconnect(self, websocket: WebSocket):
        info = self.connection_info.pop(websocket, None)
        if info:
            room_code, user_id = info
            if room_code in self.active_connections and websocket in self.active_connections[room_code]:
                self.active_connections[room_code].remove(websocket)
            remove_member(room_code, user_id)

    #Sends a message through json to all websocked connections in the specific room -> group recs 
    async def broadcast_to_room(self, room_code: str, message: dict):
        connections = self.active_connections.get(room_code, []).copy()
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print("Error sending message, disconnecting connection:", e, flush=True)
                self.disconnect(connection)
    # sends a message to a single user -> movie lists 
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
