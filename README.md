# Smart Resale ML Models 🧠💰

A sophisticated machine learning system designed to optimize resale pricing for e-commerce products. This project processes eBay listings, enriches them with Amazon catalog data (MSRP), and trains predictive models to suggest optimal resale prices.

## 🌟 Key Features

### 1. Advanced Data Enrichment 🧬
*   **Semantic Matching Engine**: Uses **TF-IDF Vectorization** and **Nearest Neighbors** to link raw eBay listings to the **1.4 million row Amazon Product Catalog**.
*   **MSRP Discovery**: Automatically finds the original "List Price" to calculate accurate depreciation.
*   **Performance**: Achieves high-confidence matching (~20% strict, higher soft) on messy user-generated titles (e.g., matching "iPad Air 2" to "Apple iPad Air 2 64GB Spgry").

### 2. Multi-Objective ML Pipeline 🤖
*   **Price Prediction**: **Random Forest** & **XGBoost** models achieving **$R^2 = 0.51$** (Doubled performance from baseline).
*   **Classification**: Identifies if a listing is "Undervalued", "Fair", or "Overpriced".
*   **Segmentation**: Clusters products into market segments using K-Means.

### 3. Production-Ready Deployment 🚀
*   **Flask Microservice**: Serves real-time predictions.
*   **Streamlit Dashboard**: Interactive UI for price checking.
*   **Smart Recommendations**: Returns actionable advice (e.g., "Increase price by €15") based on market position.

---

## 📂 Project Architecture

```
smart-resale-ml-models/
├── data/
│   ├── processed/          # Cleaned & Enriched datasets
│   └── raw/                # eBay & Amazon source files
├── models/                 # Serialized ML models (.pkl)
├── src/
│   ├── api/                # REST API Service
│   │   └── app.py          # Flask entry point
│   ├── pipeline/           # Automation Workflow
│   │   ├── run_pipeline.py     # 🏃 Master Orchestrator
│   │   ├── step1_data_prep.py  # Cleaning + Enrichment (Vector Search)
│   │   ├── step2_features.py   # Feature Engineering (Depreciation, NLP)
│   │   ├── step3_training.py   # Model Training (Regression, Classification)
│   │   └── step4_evaluation.py # Logic & Recommendations
│   └── utils/
│       └── enrichment.py       # Matching Logic
```

## 🛠️ Usage

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

Execute the full workflow (Data Cleaning -> Amazon Matching -> Training -> Evaluation):

```bash
python src/pipeline/run_pipeline.py
```

*Note: The enrichment process uses vector search and may take 1-2 minutes.*

### 3. Start the API

Launch the local prediction server:

```bash
python src/api/app.py
```
Server runs at: `http://localhost:5000`

### 4. Test Predictions

**Endpoint**: `POST /predict`

**Payload Example**:
```json
{
    "title_length": 45,
    "word_count": 8,
    "is_used": 1,
    "is_new": 0,
    "average_rating": 4.5,
    "num_reviews": 120,
    "has_brand": 1
}
```

## 📊 Performance Insights

*   **Enrichment Accuracy**: Successfully matches ~8,000+ items from the eBay sample dataset to Amazon products.
*   **Top Models**:
    *   *Regression*: Random Forest ($R^2=0.51$, RMSE=€96)
    *   *Classification*: SVC / Random Forest
    *   *Key Drivers*: Original Price, Brand (Apple/Samsung), and depreciation curve.
