# 🎬 Group Movie Recommender App

A full-stack web application that helps **small groups of users** collaboratively choose a movie to watch based on their **individual preferences**. Built using a custom PyTorch recommendation model trained on the MovieLens 100k dataset, this system dynamically combines user ratings in real time and recommends movies that best match the group's collective taste.

## 🔍 What It Does

- ✅ Users join a virtual **room** using a shared code.
- ✅ Each user is shown a **personalized list of movies** to rate (5 movies).
- ✅ After submitting ratings, the system:
  - Uses a trained **deep learning model** to predict user preferences
  - Adjusts predictions based on how each user typically scores compared to the model (bias correction)
  - Averages the predictions across the group
- ✅ Once all users are done, the backend returns a ranked list of **top 3 group movie recommendations**.

## 🧠 Model Architecture

- PyTorch neural net with:
  - Embeddings for user and item IDs
  - Dense layer for genre vector
  - Combined hidden layers with BatchNorm + Dropout
- Trained on the **full MovieLens 100k** dataset
- Evaluated with MSE loss
- Recommendation logic includes **residual bias correction**

## 🧩 Stack

| Layer      | Tech                      |
|------------|---------------------------|
| Frontend   | HTML, CSS, JavaScript (vanilla) |
| Backend    | FastAPI, Python, WebSockets |
| Model      | PyTorch                   |
| Dataset    | MovieLens 100k            |
| Testing    | Python scripts |


