let socket = null;
let roomCode = "";
let userId = "";

// Generate a random username from list 
function getRandomUsername() {
  const names = ["Falcon", "Panda", "Ghost", "Pixel", "Nova", "Zoom"];
  return names[Math.floor(Math.random() * names.length)] + Math.floor(Math.random() * 1000);
}

// Join room via websockets
//Get the room code and get rid of spaces
function joinRoom() {
  roomCode = document.getElementById("roomCodeInput").value.trim().toUpperCase();
  if (!roomCode) {
    alert("Please enter a valid room code.");
    return;
  }

  //Take in room member max 
  const userLimitInput = document.getElementById("userLimitInput");
  let expectedMembers = 3;  // Default value
  if (userLimitInput) {
    expectedMembers = parseInt(userLimitInput.value) || 3;
  }
  //assign a random user ID and update UI 
  userId = getRandomUsername();
  document.getElementById("userNameDisplay").textContent = userId;
  document.getElementById("roomCodeDisplay").textContent = roomCode;
  document.getElementById("roomJoinSection").style.display = "none";
  document.getElementById("roomInfoSection").style.display = "block";

  // Open the WebSocket connection
  socket = new WebSocket(`ws://${window.location.host}/ws`);

  socket.onopen = () => {
    // Send the join_room me ssage immediately upon connection
    console.log("WebSocket opened");
    socket.send(JSON.stringify({
      type: "join_room",
      room_code: roomCode,
      user_id: userId,
      expected_members: expectedMembers
    }));
  };

    //Handler for all messages sent by backend 
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log("Received message:", message);
    
    if (message.type === "room_joined") { 
      updateUserList(message.members);
      document.getElementById("genreSection").style.display = "block";
  
    } else if (message.type === "movie_list") {
      displayMovies(message.movies);
  
    } else if (message.type === "all_submitted") {
      document.getElementById("groupRecSection").style.display = "block";
  
    } else if (message.type === "group_recommendation") { //Most important one, outputs the picks for the group
      const recContainer = document.getElementById("groupRec");
      recContainer.innerHTML = "<h3>Group Picks:</h3><ul>" +
        message.movies.map(movie => 
          `<li>${movie.title} (${movie.score})</li>`
        ).join("") +
        "</ul>";
  
    } else if (message.type === "room_full") {
      alert(message.message);
      socket.close();
      return;
    }
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
  };

  socket.onclose = () => {
    console.log("WebSocket connection closed.");
  };
}

//shows the list of users in the room 
function updateUserList(members) {
  const userList = document.getElementById("userList");
  userList.innerHTML = "";
  members.forEach(member => {
    const li = document.createElement("li");
    li.textContent = member;
    userList.appendChild(li);
  });
}

//request movies func based off genre 
function requestMovies() {
  const genre = document.getElementById("genreDropdown").value;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({
      type: "request_movies",
      room_code: roomCode,
      genre: genre
    }));
  }
}

//display the chosen movies for rnaking 
function displayMovies(movies) {
  const movieList = document.getElementById("movieList");
  movieList.innerHTML = "";
  movies.forEach(movie => {
    const li = document.createElement("li");
    let genresText = "";
    if (movie.genres && movie.genres.length > 0) {
      genresText = `Genres: ${movie.genres.join(", ")}`;
    } else {
      genresText = "Genres: N/A";
    }
    const movieId = movie.movie_id;
    
    li.innerHTML = `
      <strong>${movie.title}</strong><br>
      ${movie.genres ? 'Genres: ' + movie.genres.join(", ") + '<br>' : ''}
      ${movie.overview ? movie.overview+ '...' : 'No description available.'}<br>
<label>Rating:
        <select id="rating-${movieId}">
          ${[...Array(11).keys()].map(i => `<option value="${i}">${i}</option>`).join('')}
        </select>
      </label>
    `;
    movieList.appendChild(li);
  });

  //Submit button to send the ratings. 
  const submitBtn = document.createElement("button");
  submitBtn.textContent = "Submit Ratings";
  submitBtn.onclick = submitRatings;
  movieList.appendChild(submitBtn);
  document.getElementById("movieListSection").style.display = "block";
}


// Send ratings to back end 
function submitRatings() {
    const items = document.querySelectorAll("#movieList li");
    const ratings = [];
  
    items.forEach(li => {
      const select = li.querySelector("select");
      if (!select) return;
  
      const movieId = select.id.replace("rating-", "");
      const rating = parseInt(select.value);
      
      ratings.push({ movie_id: movieId, rating: rating });
    });
  
    socket.send(JSON.stringify({
      type: "rating_submission",
      room_code: roomCode,
      user_id: userId,
      ratings: ratings
    }));
    
    alert("Ratings submitted! Waiting for others...");
  }



  // Get the group reccomendation 
  function triggerGroupRec() {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: "trigger_group_rec",
        room_code: roomCode
      }));
    }
  }

  //Display the final recc - for a solo movie 
  function displayGroupRecommendation(movie) {
    const groupRecDiv = document.getElementById("groupRec");
    groupRecDiv.innerHTML = `<strong>${movie.title}</strong> (Score: ${movie.score})`;

    const btn = document.getElementById("showGroupRecBtn");
    if (btn) {
      btn.style.display = "none";
    }
}