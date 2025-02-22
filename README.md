# 🎵 Music Popularity Prediction

## 📌 Project Overview

This project aims to predict the popularity of songs based on various musical features. Using machine learning techniques, we classify tracks into three categories of popularity:  
- 🧊 **Ice-cold (0-30)** → Least popular tracks
- 🔥 **Lukewarm (31-69)** → Moderately popular tracks
- 🚀 **Blazing Hot (70-100)** → Highly popular tracks

The project follows an **ETL (Extract, Transform, Load) pipeline**, performing data preprocessing, model training, and prediction, with a visualization step for insights.

## 💂️ Project Structure

```bash
music-popularity-prediction/
│── data/                      # Dataset storage
│   ├── raw/                   # Raw dataset storage
│   │   ├── tracks.csv         # Original dataset (from Kaggle)
│   ├── music_analysis.duckdb  # DuckDB database for analysis
│
│── models/                    # Trained models
│   ├── nn_model.keras         # Trained Neural Network model
│   ├── scaler.pkl             # Scaler for feature normalization
│
│── scripts/                   # Core scripts for data processing
│   ├── constants.py           # Project-wide constants
│   ├── etl_pipeline.py        # Extract, transform, and load (ETL) pipeline
│   ├── train_pipeline.py      # Model training pipeline
│   ├── predict_pipeline.py    # Prediction and visualization pipeline
│
│── requirements.txt           # Dependencies list
│── README.md                  # Project documentation

```

## 🛠 Technologies Used

This project leverages modern **data engineering** and **machine learning** tools to efficiently process and analyze music popularity. Below are the key technologies used:

### **🔹 DuckDB (Database)**
- A **fast, lightweight** columnar database optimized for analytical workloads.
- Stores processed **train** and **predict** datasets.
- Supports **SQL queries** for efficient data retrieval and transformations.

### **🔹 Prefect (Workflow Orchestration)**
- A **modern workflow management** tool for orchestrating data pipelines.
- Ensures **task dependency management**, logging, and retry handling.
- Provides a **scalable and production-ready** pipeline execution framework.

### **🔹 TensorFlow/Keras (Machine Learning)**
- Deep learning framework used for **training a neural network**.
- Implements **feature normalization**, **oversampling (SMOTE)**, and **model training**.
- Predicts **track popularity** based on various audio and musical attributes.

### **🔹 Polars & Pandas (Data Processing)**
- **Polars**: A **high-performance** DataFrame library optimized for large datasets.
- **Pandas**: Used for compatibility with `train_test_split()` and other ML tasks.
- Enables **fast transformations, filtering, and feature engineering**.

### **🔹 Matplotlib & Seaborn (Visualization)**
- **Seaborn**: Generates **data insights** through statistical plots.
- **Matplotlib**: Custom visualizations for trend analysis.
- **WordCloud**: Creates a **word cloud** of the most popular track titles.

---

📌 **With these technologies, the project ensures an efficient and scalable approach to music popularity prediction!** 🚀


## 🚀 Workflow Overview

### 📌 ETL Workflow (Extract, Transform, Load)
1️⃣ **Extract** → Downloads dataset from Kaggle *(if not already downloaded)*.  
2️⃣ **Transform** →  
   - Cleans data, removes unnecessary columns, and handles missing values.  
   - Converts **popularity** into categorical classes *(Ice-Cold, Lukewarm, Blazing Hot)*.  
   - Splits dataset into **Train** (90%) and **Predict** (10%) sets.  
3️⃣ **Load** → Stores both processed **Train** and **Predict** datasets into a **DuckDB** database.  

---

### 📌 Training Workflow  
1️⃣ **Load** → Reads the **Train** dataset from **DuckDB**.  
2️⃣ **Preprocess** →  
   - Normalizes features using **StandardScaler**.  
   - Handles **class imbalance** with **SMOTE** *(if needed)*.  
   - Splits data into **Train-Test** sets *(80% Train, 20% Test)*.  
3️⃣ **Train** → Trains a **Neural Network** classifier using **TensorFlow/Keras**.  
4️⃣ **Save** → Exports the trained model (`nn_model.keras`) and the scaler (`scaler.pkl`).  

---

### 📌 Prediction Workflow  
1️⃣ **Load** → Reads the **Predict** dataset from **DuckDB** *(new unseen tracks)*.  
2️⃣ **Preprocess** → Normalizes features using the saved **StandardScaler**.  
3️⃣ **Predict** → Uses the trained **Neural Network** model to classify new tracks into popularity classes.  
4️⃣ **Store Results** → Updates the **DuckDB** table with the predicted popularity classes.  
5️⃣ **Visualize** →  
   - **Class Distribution** → Shows the number of tracks per popularity class.  
   - **Danceability vs Energy Scatter Plot** → Highlights trends across different classes.  
   - **Word Cloud (Hits Only)** → Generates a word cloud with only the most popular track titles.  

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

## 💻 System Requirements

To run this project efficiently, ensure that your system meets the following requirements:

### **🔹 Hardware Requirements**
- 💾 **RAM**: At least **8GB** (16GB recommended for large datasets)
- 💽 **Disk Space**: At least **2GB** of free storage
- 🔥 **GPU (Optional)**: Recommended for faster model training with TensorFlow

### **🔹 Software Requirements**
- 🐍 **Python 3.8+** (recommended: **3.10+**)
- 💽 **Operating System**: Windows, macOS, or Linux
- 📦 **Required Libraries**: Install using `requirements.txt`  
  ```bash
  pip install -r requirements.txt


## 🛠️ Installation & Setup

### Clone the repository

git clone <https://github.com/ngarts/music-popularity-prediction.git>

cd music-popularity-prediction

### Create and activate virtual environment

python -m venv venv

source venv/bin/activate # On Windows: venv\\Scripts\\activate

### Install dependencies

pip install -r requirements.txt

## 🔑 Kaggle API Setup

To download datasets from Kaggle, you need to configure your **Kaggle API key**.

### **1️⃣ Download your `kaggle.json` API key**
1. Go to your **[Kaggle Account Settings](https://www.kaggle.com/settings)**.
2. Scroll down to **API** and click **"Create New API Token"**.
3. The file **`kaggle.json`** will be downloaded automatically.

### **2️⃣ Move `kaggle.json` to the correct location**
Move the downloaded file to your Kaggle API directory:

#### **On Windows (PowerShell)**
```powershell
mkdir C:\Users\$env:USERNAME\.kaggle -ErrorAction SilentlyContinue
Move-Item -Path "$HOME\Downloads\kaggle.json" -Destination "$HOME\.kaggle\"
```
#### **On Linux/macOS (Bash)**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
```

### **3️⃣ Set Correct Permissions**
To ensure the file is secure, set proper read permissions:

#### **On Linux/macOS**
```bash
chmod 600 ~/.kaggle/kaggle.json
```
#### **On Windows (PowerShell)**
```powershell
icacls "$HOME\.kaggle\kaggle.json" /grant %USERNAME%:R
```

### **4️⃣ Verify Installation**
To check if the API key is working, run:
```bash
kaggle datasets list
```
✅ If you see a list of datasets, your Kaggle API is configured correctly!

## 🚀 How to Run the Pipelines

### 1️⃣ Retrieve the dataset from kaggle 

```bash
python scripts/etl_pipeline.py
```

### 2️⃣ Train the model

```bash
python scripts/train_pipeline.py
```

### 3️⃣ Predict popularity for new songs

```bash
python scripts/predict_pipeline.py
```

### 📡 Running Prefect Server & Monitoring Pipelines

Prefect provides a **web-based dashboard** to **monitor and manage workflows** in real time. Follow these steps to start the Prefect server and track your pipeline executions.

#### **1️⃣ Start Prefect Orion Server**
The **Orion server** is Prefect’s UI for monitoring workflows. Run the following command:

```bash
prefect server start
```

Once started, the Prefect dashboard will be accessible at:

🔗 http://127.0.0.1:4200

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