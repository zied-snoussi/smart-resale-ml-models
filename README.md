# Smart Resale ML Models

A machine learning project designed to analyze and predict resale prices for e-commerce products (specifically tailored for eBay data). This system provides price predictions, pricing classification (undervalued/overvalued), and optimal pricing recommendations.

## 🚀 Features

The application fulfills three main objectives:

1.  **Resale Price Prediction (Regression)**: Predicts the estimated resale price of a product based on its features (condition, brand, reviews, specifications).
2.  **Pricing Classification**: Evaluates if a given price is "Undervalued", "Fair Price", or "Overpriced" compared to market trends.
3.  **Smart Recommendations**: Suggests pricing strategies and optimal price points to maximize sales or profit.
4.  **Clustering Analysis**: Groups similar products to identify market segments.

## 📂 Project Structure

```
smart-resale-ml-models/
├── data/
│   ├── processed/          # Processed datasets for training
│   └── raw/                # Original datasets (eBay, Amazon)
├── models/                 # Saved trained models (.pkl files)
├── notebooks/              # Jupyter notebooks for exploration
├── src/
│   ├── app.py              # Flask API for serving predictions
│   ├── train_complete.py   # Main pipeline script to train all models
│   ├── models_complete.py  # Model classes (Regression, Classification, etc.)
│   ├── preprocessing.py    # Feature engineering and cleaning logic
│   ├── predict.py          # Standalone prediction script
│   └── ...
├── requirements.txt        # Python dependencies
└── README.md
```

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd smart-resale-ml-models
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### 1. Training the Models

To train the entire pipeline (processing data, training regression, classification, and clustering models), run:

```bash
python src/train_complete.py
```

This will:
*   Load data from `data/raw/`
*   Preprocess and extract features
*   Train multiple models and select the best ones
*   Save the trained models to the `models/` directory:
    *   `ebay_price_predictor.pkl`
    *   `ebay_pricing_classifier.pkl`
    *   `ebay_recommender.pkl`
    *   `ebay_clustering.pkl`

### 2. Running the API

Start the Flask API server to serve predictions:

```bash
python src/app.py
```

The server will start at `http://localhost:5000`.

### 3. Making Predictions (API)

**Endpoint:** `POST /predict`

**Example Request:**
```json
{
    "average_rating": 4.5,
    "num_reviews": 150,
    "title_length": 45,
    "word_count": 8,
    "is_new": 0,
    "is_used": 1,
    "is_refurbished": 0,
    "has_brand": 1,
    "screen_size": 6.1,
    "memory_gb": 128,
    "current_price": 550
}
```

**Example Response:**
```json
{
    "predicted_price": 540.50,
    "pricing_category": "Juste prix",
    "recommendations": [
        {
            "action": "Maintain Price",
            "suggested_price": 540.50,
            "reason": "Current price is improving market competitiveness."
        }
    ]
}
```

### 4. Health Check

**Endpoint:** `GET /health`
Returns the status of the API and loaded models.

## 📊 Models Used

*   **Regression:** Random Forest, Gradient Boosting, Linear Regression, SVR.
*   **Classification:** Random Forest Classifier, Logistic Regression, SVC.
*   **Clustering:** K-Means, DBSCAN.

## 📝 Features Used

The models are trained on features such as:
*   Product Condition (New, Used, Refurbished)
*   Review Metrics (Average Rating, Number of Reviews)
*   Text Features (Title Length, Word Count)
*   Specifications (Screen Size, Memory, Brand presence)
