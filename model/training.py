import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import joblib
# ========= PREPROCESSING =========
# --- FILE PATHS ---
base_path = "data/ml-100k/ua.base"
test_path = "data/ml-100k/ua.test"  
items_path = "data/ml-100k/u.item"
genres_path = "data/ml-100k/u.genre"

# --- LOAD DATA ---
ua_base = pd.read_csv(base_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
ua_test = pd.read_csv(test_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

items_df = pd.read_csv(items_path, sep='|', header=None, encoding='latin-1')
genres_df = pd.read_csv(genres_path, sep='|', names=['genre', 'genre_id'], header=None)

# --- SETUP GENRE COLUMNS ---
genre_cols = genres_df['genre'].tolist()
movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_cols
items_df = items_df.iloc[:, :len(movie_columns)]
items_df.columns = movie_columns

# --- MERGE GENRES INTO BASE SET ---
ua_base_merged = pd.merge(ua_base, items_df, left_on='item_id', right_on='movie_id')

# --- ENCODE USERS AND ITEMS ---
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

ua_base_merged['user_idx'] = user_encoder.fit_transform(ua_base_merged['user_id'])
ua_base_merged['item_idx'] = item_encoder.fit_transform(ua_base_merged['item_id'])
joblib.dump(user_encoder, "user_encoder.pkl")
joblib.dump(item_encoder, "item_encoder.pkl")
print("Saved encoders")

# --- FILTER TEST SET TO SEEN ITEMS ONLY ---
seen_items = ua_base['item_id'].unique()
ua_test_filtered = ua_test[ua_test['item_id'].isin(seen_items)]

# --- MERGE GENRES INTO TEST SET ---
ua_test_merged = pd.merge(ua_test_filtered, items_df, left_on='item_id', right_on='movie_id')
ua_test_merged['user_idx'] = user_encoder.transform(ua_test_merged['user_id'])
ua_test_merged['item_idx'] = item_encoder.transform(ua_test_merged['item_id'])

# --- SAVE PROCESSED CSVs ---
train_df = ua_base_merged[['user_idx', 'item_idx', 'rating'] + genre_cols]
test_df = ua_test_merged[['user_idx', 'item_idx', 'rating'] + genre_cols]

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
print("Saved train.csv and test.csv")

# ========= PYTORCH DATASET =========
class MovieLensDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.users = torch.tensor(self.df['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(self.df['item_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(self.df['rating'].values, dtype=torch.float32)
        # All remaining columns (from column index 3 onward) are genre features
        self.genres = torch.tensor(self.df.iloc[:, 3:].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'item': self.items[idx],
            'genre': self.genres[idx],
            'rating': self.ratings[idx]
        }

# Create datasets and dataloaders
train_dataset = MovieLensDataset("train.csv")
test_dataset = MovieLensDataset("test.csv")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ========= MODEL DEFINITION =========
class MovieRecModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres, emb_dim=32):
        super(MovieRecModel, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.genre_fc = nn.Linear(num_genres, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 3, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, user, item, genre):
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)
        genre_vec = self.genre_fc(genre)
        x = torch.cat([user_vec, item_vec, genre_vec], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x.squeeze()


# ========= SETUP MODEL, LOSS, OPTIMIZER =========
num_users = train_df['user_idx'].nunique()
num_items = train_df['item_idx'].nunique()
num_genres = len(genre_cols)

model = MovieRecModel(num_users, num_items, num_genres, emb_dim=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========= TRAINING LOOP =========
num_epochs = 250
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['user'], batch['item'], batch['genre'])
        loss = criterion(outputs, batch['rating'])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch['user'].size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['user'], batch['item'], batch['genre'])
            loss = criterion(outputs, batch['rating'])
            val_loss += loss.item() * batch['user'].size(0)
    val_loss /= len(test_dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# ========= EVALUATION ON TEST SET =========
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch['user'], batch['item'], batch['genre'])
        loss = criterion(outputs, batch['rating'])
        test_loss += loss.item() * batch['user'].size(0)

test_loss /= len(test_dataset)
print(f"Test Loss: {test_loss:.4f}")

# ========= SAVE MODEL =========
torch.save(model.state_dict(), "model.pth")
print(" Saved model as model.pth")
