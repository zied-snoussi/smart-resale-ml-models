from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, classification_report, silhouette_score)
import numpy as np
import pandas as pd

class RegressionTrainer:
    """OBJECTIF 1: Prédire le prix de revente"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf')
        }
        self.results = {}
        self.best_model = None
    
    def train_all(self, X_train, y_train, X_test, y_test):
        """Train all regression models"""
        print(f"\n{'='*60}")
        print(f"🎯 OBJECTIF 1: PRÉDICTION DU PRIX DE REVENTE")
        print(f"{'='*60}\n")
        
        for name, model in self.models.items():
            print(f"📊 Training {name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"   RMSE: €{rmse:.2f}")
            print(f"   MAE: €{mae:.2f}")
            print(f"   R²: {r2:.4f}\n")
        
        best_name = min(self.results, key=lambda x: self.results[x]['rmse'])
        self.best_model = self.results[best_name]['model']
        print(f"🏆 Best Model: {best_name}")
        
        return self.results

class ClassificationTrainer:
    """OBJECTIF 2: Évaluer si un produit est surévalué/sous-évalué"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVC': SVC(kernel='rbf', probability=True)
        }
        self.results = {}
        self.best_model = None
    
    def create_price_categories(self, y_pred, y_true):
        """Create categories: Sous-évalué / Juste prix / Surévalué"""
        diff = (y_pred - y_true) / y_true * 100
        
        categories = np.zeros(len(diff))
        categories[diff < -10] = 0  # Sous-évalué
        categories[(diff >= -10) & (diff <= 10)] = 1  # Juste prix
        categories[diff > 10] = 2  # Surévalué
        
        return categories.astype(int)
    
    def train_all(self, X_train, y_train_cat, X_test, y_test_cat):
        """Train all classification models"""
        print(f"\n{'='*60}")
        print(f"🏷️ OBJECTIF 2: CLASSIFICATION DU PRICING")
        print(f"{'='*60}\n")
        
        for name, model in self.models.items():
            print(f"📊 Training {name}...")
            
            model.fit(X_train, y_train_cat)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test_cat, y_pred)
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"   Accuracy: {accuracy:.4f}\n")
        
        best_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        self.best_model = self.results[best_name]['model']
        print(f"🏆 Best Model: {best_name}")
        
        return self.results

class RecommendationSystem:
    """OBJECTIF 3: Suggérer le meilleur prix et timing"""
    
    def __init__(self, regression_model):
        self.model = regression_model
    
    def suggest_optimal_price(self, X, current_price):
        """Suggérer le meilleur prix de mise en vente"""
        predicted_price = self.model.predict(X)[0]
        
        recommendations = []
        
        # Analyse de la situation
        diff_pct = ((current_price - predicted_price) / predicted_price) * 100
        
        if diff_pct > 15:
            recommendations.append({
                'action': 'Baisser le prix',
                'suggested_price': predicted_price * 1.05,
                'reason': f'Prix actuel {diff_pct:.1f}% trop élevé',
                'impact': 'Vente rapide (< 7 jours)'
            })
        elif diff_pct < -10:
            recommendations.append({
                'action': 'Augmenter le prix',
                'suggested_price': predicted_price * 0.95,
                'reason': f'Prix actuel {abs(diff_pct):.1f}% trop bas',
                'impact': 'Maximiser le profit'
            })
        else:
            recommendations.append({
                'action': 'Prix optimal',
                'suggested_price': current_price,
                'reason': 'Prix aligné avec le marché',
                'impact': 'Vente en 10-15 jours'
            })
        
        return recommendations

class ClusteringAnalysis:
    """OBJECTIF 4: Segmenter les produits par comportement de dépréciation"""
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_profiles = {}
    
    def fit(self, X):
        """Cluster products"""
        print(f"\n{'='*60}")
        print(f"📦 OBJECTIF 4: CLUSTERING & SEGMENTATION")
        print(f"{'='*60}\n")
        
        self.model.fit(X)
        labels = self.model.labels_
        
        score = silhouette_score(X, labels)
        print(f"Silhouette Score: {score:.4f}\n")
        
        # Analyze clusters
        for i in range(self.n_clusters):
            cluster_data = X[labels == i]
            self.cluster_profiles[i] = {
                'size': len(cluster_data),
                'avg_price': cluster_data[:, 0].mean() if X.shape[1] > 0 else 0,
                'characteristics': f'Cluster {i+1}'
            }
            print(f"Cluster {i+1}: {len(cluster_data)} products")
        
        return labels

if __name__ == "__main__":
    # Test with dummy data
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    
    from preprocessing import prepare_train_test_split
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)
    
    # Test regression
    reg_trainer = RegressionTrainer()
    reg_trainer.train_all(X_train, y_train, X_test, y_test)
    
    print("\n✅ Models module test complete!")