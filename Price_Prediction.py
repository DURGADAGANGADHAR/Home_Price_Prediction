import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
import joblib
import os
from datetime import datetime
import random

warnings.filterwarnings('ignore')

class HomePriceDataGenerator:
    """Generate synthetic home price data for demonstration"""
    
    def __init__(self, n_samples=5000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
    
    def generate_data(self):
        """Generate comprehensive synthetic home price dataset"""
        print("Generating synthetic home price data...")
        
        # Basic house features
        bedrooms = np.random.choice([2, 3, 4, 5, 6], size=self.n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05])
        bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], size=self.n_samples, 
                                   p=[0.05, 0.1, 0.3, 0.2, 0.2, 0.1, 0.05])
        
        # Square footage (correlated with bedrooms)
        sqft_base = bedrooms * 300 + np.random.normal(800, 200, self.n_samples)
        square_feet = np.clip(sqft_base, 500, 5000)
        
        # Lot size
        lot_size = np.random.gamma(2, 2000) + 2000
        lot_size = np.clip(lot_size, 2000, 20000)
        
        # Age of house
        house_age = np.random.exponential(15)
        house_age = np.clip(house_age, 0, 100)
        
        # Location features
        neighborhoods = ['Downtown', 'Suburbs', 'Rural', 'Waterfront', 'Hills', 'Industrial']
        neighborhood = np.random.choice(neighborhoods, size=self.n_samples, 
                                      p=[0.15, 0.4, 0.15, 0.1, 0.15, 0.05])
        
        # Distance features (in miles)
        distance_to_downtown = np.random.exponential(8) + 1
        distance_to_downtown = np.clip(distance_to_downtown, 1, 30)
        
        # School ratings (1-10 scale)
        school_rating = np.random.beta(3, 2) * 10
        school_rating = np.clip(school_rating, 1, 10)
        
        # Distance to schools (in miles)
        distance_to_school = np.random.exponential(2) + 0.5
        distance_to_school = np.clip(distance_to_school, 0.5, 10)
        
        # Hospital proximity
        distance_to_hospital = np.random.exponential(5) + 1
        distance_to_hospital = np.clip(distance_to_hospital, 1, 25)
        
        # Crime rate (per 1000 residents)
        crime_rate = np.random.gamma(2, 5)
        crime_rate = np.clip(crime_rate, 1, 50)
        
        # Income level of neighborhood (median household income in thousands)
        income_level = np.random.normal(65, 25)
        income_level = np.clip(income_level, 25, 150)
        
        # Additional amenities
        has_garage = np.random.choice([0, 1], size=self.n_samples, p=[0.3, 0.7])
        has_pool = np.random.choice([0, 1], size=self.n_samples, p=[0.8, 0.2])
        has_basement = np.random.choice([0, 1], size=self.n_samples, p=[0.6, 0.4])
        has_fireplace = np.random.choice([0, 1], size=self.n_samples, p=[0.7, 0.3])
        
        # Property type
        property_types = ['Single Family', 'Condo', 'Townhouse', 'Multi Family']
        property_type = np.random.choice(property_types, size=self.n_samples, 
                                       p=[0.6, 0.2, 0.15, 0.05])
        
        # Market conditions
        market_trend = np.random.normal(1, 0.1, self.n_samples)  # Market multiplier
        
        # Calculate price based on features with realistic relationships
        base_price = (
            square_feet * 80 +  # Base price per sqft
            bedrooms * 15000 +  # Premium for more bedrooms
            bathrooms * 8000 +  # Premium for more bathrooms
            lot_size * 2 +  # Land value
            school_rating * 8000 +  # School quality premium
            income_level * 500 +  # Neighborhood income effect
            has_garage * 15000 +  # Garage premium
            has_pool * 25000 +  # Pool premium
            has_basement * 12000 +  # Basement premium
            has_fireplace * 8000  # Fireplace premium
        )
        
        # Neighborhood adjustments
        neighborhood_multipliers = {
            'Downtown': 1.3,
            'Suburbs': 1.0,
            'Rural': 0.7,
            'Waterfront': 1.8,
            'Hills': 1.2,
            'Industrial': 0.6
        }
        
        neighborhood_adjustment = np.array([neighborhood_multipliers[n] for n in neighborhood])
        
        # Distance penalties
        distance_penalty = (
            distance_to_downtown * -500 +
            distance_to_school * -2000 +
            distance_to_hospital * -300
        )
        
        # Crime rate penalty
        crime_penalty = crime_rate * -1000
        
        # Age depreciation
        age_penalty = house_age * -800
        
        # Property type adjustment
        property_type_multipliers = {
            'Single Family': 1.0,
            'Condo': 0.8,
            'Townhouse': 0.9,
            'Multi Family': 1.1
        }
        
        property_adjustment = np.array([property_type_multipliers[pt] for pt in property_type])
        
        # Final price calculation
        price = (
            (base_price + distance_penalty + crime_penalty + age_penalty) * 
            neighborhood_adjustment * 
            property_adjustment * 
            market_trend
        )
        
        # Add some noise and ensure positive prices
        price = price * np.random.normal(1, 0.1, self.n_samples)
        price = np.clip(price, 50000, 2000000)
        
        # Create DataFrame
        data = pd.DataFrame({
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'square_feet': square_feet.astype(int),
            'lot_size': lot_size.astype(int),
            'house_age': house_age.round(1),
            'neighborhood': neighborhood,
            'distance_to_downtown': distance_to_downtown.round(2),
            'school_rating': school_rating.round(1),
            'distance_to_school': distance_to_school.round(2),
            'distance_to_hospital': distance_to_hospital.round(2),
            'crime_rate': crime_rate.round(1),
            'income_level': income_level.round(0).astype(int),
            'has_garage': has_garage,
            'has_pool': has_pool,
            'has_basement': has_basement,
            'has_fireplace': has_fireplace,
            'property_type': property_type,
            'price': price.round(0).astype(int)
        })
        
        print(f"Generated {len(data)} home records with {len(data.columns)} features")
        return data

class HomePricePredictor:
    """XGBoost-based home price prediction system"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_trained = False
        
    def load_data(self, data_source=None):
        """Load data from file or generate synthetic data"""
        if data_source and os.path.exists(data_source):
            print(f"Loading data from {data_source}")
            self.data = pd.read_csv(data_source)
        else:
            print("Generating synthetic home price data...")
            generator = HomePriceDataGenerator()
            self.data = generator.generate_data()
        
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        # Basic info
        print("\nDataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Summary statistics
        print("\nSummary Statistics:")
        print(self.data.describe())
        
        # Missing values
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing Values:")
            print(missing[missing > 0])
        else:
            print("\nNo missing values found!")
        
        # Price statistics
        print(f"\nPrice Statistics:")
        print(f"Mean Price: ${self.data['price'].mean():,.0f}")
        print(f"Median Price: ${self.data['price'].median():,.0f}")
        print(f"Price Range: ${self.data['price'].min():,.0f} - ${self.data['price'].max():,.0f}")
        print(f"Standard Deviation: ${self.data['price'].std():,.0f}")
        
        # Correlation with price
        print("\nFeatures most correlated with price:")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlations = self.data[numeric_cols].corr()['price'].abs().sort_values(ascending=False)
        print(correlations.head(10))
        
    def visualize_data(self):
        """Create visualizations for data analysis"""
        print("\nCreating data visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Home Price Data Analysis', fontsize=16, fontweight='bold')
        
        # Price distribution
        axes[0, 0].hist(self.data['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].ticklabel_format(style='plain', axis='x')
        
        # Price vs Square Feet
        axes[0, 1].scatter(self.data['square_feet'], self.data['price'], alpha=0.6, color='coral')
        axes[0, 1].set_title('Price vs Square Feet')
        axes[0, 1].set_xlabel('Square Feet')
        axes[0, 1].set_ylabel('Price ($)')
        
        # Price by neighborhood
        neighborhood_prices = self.data.groupby('neighborhood')['price'].mean().sort_values(ascending=True)
        axes[0, 2].barh(range(len(neighborhood_prices)), neighborhood_prices.values, color='lightgreen')
        axes[0, 2].set_title('Average Price by Neighborhood')
        axes[0, 2].set_xlabel('Average Price ($)')
        axes[0, 2].set_yticks(range(len(neighborhood_prices)))
        axes[0, 2].set_yticklabels(neighborhood_prices.index)
        
        # School rating vs Price
        axes[1, 0].scatter(self.data['school_rating'], self.data['price'], alpha=0.6, color='gold')
        axes[1, 0].set_title('Price vs School Rating')
        axes[1, 0].set_xlabel('School Rating')
        axes[1, 0].set_ylabel('Price ($)')
        
        # Crime rate vs Price
        axes[1, 1].scatter(self.data['crime_rate'], self.data['price'], alpha=0.6, color='salmon')
        axes[1, 1].set_title('Price vs Crime Rate')
        axes[1, 1].set_xlabel('Crime Rate (per 1000)')
        axes[1, 1].set_ylabel('Price ($)')
        
        # Correlation heatmap
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 2].set_title('Correlation Matrix')
        axes[1, 2].set_xticks(range(len(numeric_cols)))
        axes[1, 2].set_yticks(range(len(numeric_cols)))
        axes[1, 2].set_xticklabels(numeric_cols, rotation=45, ha='right')
        axes[1, 2].set_yticklabels(numeric_cols)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
        
        plt.tight_layout()
        plt.show()
        
    def preprocess_data(self):
        """Preprocess data for modeling"""
        print("\nPreprocessing data...")
        
        # Separate features and target
        X = self.data.drop('price', axis=1).copy()
        y = self.data['price'].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Feature engineering
        X['price_per_sqft'] = y / X['square_feet']
        X['room_ratio'] = X['bathrooms'] / X['bedrooms']
        X['age_category'] = pd.cut(X['house_age'], bins=[0, 5, 15, 30, 100], 
                                 labels=['New', 'Recent', 'Mature', 'Old'])
        X['age_category'] = LabelEncoder().fit_transform(X['age_category'])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Final feature set: {len(self.feature_names)} features")
        print(f"Features: {self.feature_names}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2):
        """Train XGBoost model with hyperparameter tuning"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define XGBoost model
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Hyperparameter tuning
        print("\nPerforming hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score (RMSE): ${np.sqrt(-grid_search.best_score_):,.0f}")
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.X_train = X_train_scaled
        self.y_train = y_train
        
        self.is_trained = True
        print("Model training completed!")
        
    def evaluate_model(self):
        """Evaluate model performance"""
        if not self.is_trained:
            print("Model not trained yet!")
            return
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Training metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        # Test metrics
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print("Performance Metrics:")
        print(f"Training RMSE: ${train_rmse:,.0f}")
        print(f"Training MAE: ${train_mae:,.0f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test RMSE: ${test_rmse:,.0f}")
        print(f"Test MAE: ${test_mae:,.0f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Overfitting check
        if train_rmse < test_rmse * 0.8:
            print("\n⚠️  Potential overfitting detected!")
        else:
            print("\n✓ Model shows good generalization")
        
        # Feature importance
        self.plot_feature_importance()
        
        # Prediction vs Actual plot
        self.plot_predictions(y_test_pred)
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('XGBoost Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        print("\nTop 10 Most Important Features:")
        top_features = feature_importance.tail(10)
        for idx, row in top_features.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    
    def plot_predictions(self, y_pred):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, y_pred, alpha=0.6, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title('Actual vs Predicted Prices')
        plt.ticklabel_format(style='plain')
        
        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = self.y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price ($)')
        plt.ylabel('Residuals ($)')
        plt.title('Residual Plot')
        plt.ticklabel_format(style='plain')
        
        plt.tight_layout()
        plt.show()
    
    def predict_price(self, features):
        """Predict price for new house features"""
        if not self.is_trained:
            print("Model not trained yet!")
            return None
        
        # Convert to DataFrame if dictionary
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # Encode categorical variables
        for col in features_df.columns:
            if col in self.label_encoders:
                if features_df[col].iloc[0] in self.label_encoders[col].classes_:
                    features_df[col] = self.label_encoders[col].transform(features_df[col])
                else:
                    # Handle unknown categories
                    features_df[col] = 0
        
        # Add engineered features
        if 'square_feet' in features_df.columns:
            features_df['price_per_sqft'] = 0  # Will be calculated after prediction
            features_df['room_ratio'] = features_df['bathrooms'] / features_df['bedrooms']
            
            # Age category
            age = features_df['house_age'].iloc[0]
            if age <= 5:
                features_df['age_category'] = 0  # New
            elif age <= 15:
                features_df['age_category'] = 1  # Recent
            elif age <= 30:
                features_df['age_category'] = 2  # Mature
            else:
                features_df['age_category'] = 3  # Old
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Reorder columns to match training data
        features_df = features_df[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction
    
    def save_model(self, filename='home_price_model.pkl'):
        """Save trained model"""
        if not self.is_trained:
            print("No trained model to save!")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='home_price_model.pkl'):
        """Load trained model"""
        if not os.path.exists(filename):
            print(f"Model file {filename} not found!")
            return
        
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {filename}")

def main():
    """Main function to run the home price prediction system"""
    print("="*70)
    print("           XGBOOST HOME PRICE PREDICTION SYSTEM")
    print("="*70)
    
    # Initialize predictor
    predictor = HomePricePredictor()
    
    # Load and explore data
    data = predictor.load_data()
    predictor.explore_data()
    predictor.visualize_data()
    
    # Preprocess and train
    X, y = predictor.preprocess_data()
    predictor.train_model(X, y)
    
    # Evaluate model
    metrics = predictor.evaluate_model()
    
    # Save model
    predictor.save_model()
    
    # Example predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Example house 1: Luxury home
    luxury_house = {
        'bedrooms': 5,
        'bathrooms': 3.5,
        'square_feet': 3500,
        'lot_size': 8000,
        'house_age': 8,
        'neighborhood': 'Waterfront',
        'distance_to_downtown': 12,
        'school_rating': 9.2,
        'distance_to_school': 1.5,
        'distance_to_hospital': 4,
        'crime_rate': 5,
        'income_level': 95,
        'has_garage': 1,
        'has_pool': 1,
        'has_basement': 1,
        'has_fireplace': 1,
        'property_type': 'Single Family'
    }
    
    # Example house 2: Starter home
    starter_house = {
        'bedrooms': 3,
        'bathrooms': 2,
        'square_feet': 1800,
        'lot_size': 5000,
        'house_age': 15,
        'neighborhood': 'Suburbs',
        'distance_to_downtown': 8,
        'school_rating': 7.5,
        'distance_to_school': 2,
        'distance_to_hospital': 6,
        'crime_rate': 12,
        'income_level': 65,
        'has_garage': 1,
        'has_pool': 0,
        'has_basement': 0,
        'has_fireplace': 1,
        'property_type': 'Single Family'
    }
    
    # Make predictions
    luxury_price = predictor.predict_price(luxury_house)
    starter_price = predictor.predict_price(starter_house)
    
    print(f"\nLuxury Waterfront Home Prediction: ${luxury_price:,.0f}")
    print("Features: 5 bed, 3.5 bath, 3500 sqft, waterfront, excellent schools")
    
    print(f"\nStarter Suburban Home Prediction: ${starter_price:,.0f}")
    print("Features: 3 bed, 2 bath, 1800 sqft, suburbs, good schools")
    
    # Model insights
    print("\n" + "="*60)
    print("MODEL INSIGHTS")
    print("="*60)
    
    print("Key factors affecting home prices:")
    print("1. Square footage - Larger homes command higher prices")
    print("2. Location - Waterfront and hills premium, industrial discount")
    print("3. School quality - Higher rated schools increase property values")
    print("4. Crime rates - Lower crime areas have higher property values")
    print("5. Neighborhood income - Wealthier areas have higher home prices")
    print("6. Amenities - Pools, garages, basements add significant value")
    print("7. Age - Newer homes generally worth more")
    print("8. Distance to amenities - Close to schools and hospitals preferred")
    
    print(f"\nModel Performance Summary:")
    print(f"Test R² Score: {metrics['test_r2']:.4f}")
    print(f"Test RMSE: ${metrics['test_rmse']:,.0f}")
    print("The model explains {:.1f}% of price variation".format(metrics['test_r2'] * 100))

if __name__ == "__main__":
    main()
