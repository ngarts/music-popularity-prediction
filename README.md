# 🎵 Music Popularity Prediction

## 📌 Project Overview

This project aims to predict the popularity of songs based on various musical features. Using machine learning techniques, we classify tracks into three categories of popularity:  
- 🧊 **Ice-cold (0-30)** → Least popular tracks
- 🔥 **Lukewarm (31-69)** → Moderately popular tracks
- 🚀 **Blazing Hot (70-100)** → Highly popular tracks

The project follows an **ETL (Extract, Transform, Load) pipeline**, performing data preprocessing, model training, and prediction, with a visualization step for insights.

## 💂️ Project Structure

```bash
my_project/
│── data/                 # Dataset storage
│   ├── predict/          # Predict Dataset storage
│   │   ├── new_tracks.csv  # Tracks for prediction
│
│── models/               # Trained models
│   ├── nn_model.keras      # Trained Neural Network model
│   ├── scaler.pkl          # Scaler for feature normalization
│
│── scripts/              # Core scripts for data processing
│   ├── extract.py          # Extracts dataset from Kaggle
│   ├── transform.py        # Cleans and preprocesses data
│   ├── load.py             # Loads data into DuckDB
│   ├── train.py            # Trains the neural network
│   ├── predict.py          # Predicts popularity of new tracks
│   ├── visualize.py        # Generates visual insights
│   ├── train_pl.py         # Starts pipeline to train the model
│   ├── predict_pl.py       # Starts pipeline to predict popularity for new songs
│
│── requirements.txt      # Dependencies
│── README.md            # Project documentation
```

## 🚀 Workflows

### Training workflow

1️⃣ **Extract** → Downloads dataset from Kaggle.  
2️⃣ **Transform** → Cleans data, removes unnecessary columns, and normalizes values.  
3️⃣ **Load** → Stores processed data into a **DuckDB** database.  
4️⃣ **Train** → Trains a **Neural Network** classifier (Keras) to predict popularity. 

### Predict workflow

1️⃣ **Extract** → Retrieve local dataset with new songs.  
2️⃣ **Transform** → Cleans data, removes unnecessary columns, and normalizes values.  
3️⃣ **Load** → Stores processed data into a **DuckDB** database.   
5️⃣ **Predict** → Uses the trained model to classify new tracks.  
6️⃣ **Visualize** → Generates insights via **word clouds, scatter plots, and histograms**.

## 📊 Dataset & Features

### 🛠️ Source

The dataset used for training comes from **Kaggle**:

```sh
kaggle datasets download -d yamaerenay/spotify-dataset-19212020-600k-tracks -p data/ --unzip
```

It contains **over 600,000 tracks** with metadata and extracted audio features.

### 🔎 Selected Features

| Feature | Description |
| --- | --- |
| **danceability** | How suitable a track is for dancing (0-1) |
| **energy** | Intensity and activity level (0-1) |
| **tempo** | Beats per minute (BPM) |
| **valence** | Positivity of the track (0-1) |
| **loudness** | Overall volume in decibels (dB) |
| **speechiness** | Presence of spoken words (0-1) |
| **instrumentalness** | Likelihood of a track being instrumental (0-1) |
| **acousticness** | Acoustic properties (0-1) |
| **mode** | Key mode (0=Minor, 1=Major) |
| **key** | Musical key (0=C, 1=C#, ..., 11=B) |
| **duration_ms** | Duration in milliseconds |

## 🧠 Model Training & Challenges

### ⚙️ ****Neural Network Architecture****

- **3-layer Dense Neural Network**
- Activation: **ReLU** in hidden layers, **Softmax** in output
- Optimizer: **Adam**
- Loss function: **Categorical Crossentropy**
- Balanced **class weighting** to account for dataset imbalance

### 🏆 ****Challenges & Solutions****

| Issue | Solution |
| --- | --- |
| **Data Imbalance** | Used compute_class_weight() to balance class distribution |
| **Overfitting** | Implemented **Dropout layers** (30%, 20%) and **EarlyStopping** |
| **Normalization Issue** | Ensured StandardScaler() was saved and applied consistently in both training and prediction |

## 🛠️ Installation & Setup

\# Clone the repository

git clone <https://github.com/ngarts/music-popularity-prediction.git>

cd music-popularity-prediction

\# Create and activate virtual environment

python -m venv venv

source venv/bin/activate # On Windows: venv\\Scripts\\activate

\# Install dependencies

pip install -r requirements.txt

## 🚀 How to Run the Pipelines

### 1️⃣ Train the model

python scripts/train_pl.py

### 2️⃣ Predict popularity for new songs

python scripts/predict_pl.py

## 📈 Results & Visualizations

- The model achieves **~67% accuracy** on validation data.
- **Insights**:
  - Danceability and energy show **some correlation** with popularity.
  - High valence tracks are **more likely** to be in **Blazing Hot**.
  - Word cloud analysis shows top **recurring words** in **popular songs**.

### 📊 ****Sample Visualizations****

## 🤝 Contributions

Feel free to **fork** this repository, submit **pull requests**, or suggest improvements! 🚀🎶

## 🐟 License

This project is licensed under the **MIT License**.