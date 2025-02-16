import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta
from sklearn.utils.class_weight import compute_class_weight

class PalletPredictiveSystem:
    def __init__(self):
        self.deviation_model = None
        self.eta_model = None
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
    def prepare_features(self, data, for_training=True):
        """Create advanced features for prediction"""
        # Reset index to handle cases where timestamp might be in the index
        data = data.reset_index(drop=True).copy()  # Create a copy to avoid SettingWithCopyWarning
        features = pd.DataFrame()
        
        # Time-based features with cyclical encoding
        timestamps = pd.to_datetime(data['timestamp'])
        features['hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24.0)
        features['hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24.0)
        features['day_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7.0)
        features['day_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7.0)
        features['month_sin'] = np.sin(2 * np.pi * timestamps.dt.month / 12.0)
        features['month_cos'] = np.cos(2 * np.pi * timestamps.dt.month / 12.0)
        
        # Calculate speed and acceleration using time differences
        data = data.sort_values(['Pallet_ID', 'timestamp'])
        data['time_diff'] = data.groupby('Pallet_ID')['timestamp'].diff().dt.total_seconds() / 3600
        
        # Calculate distances between consecutive points
        data['lat_diff'] = data.groupby('Pallet_ID')['latitude'].diff()
        data['lon_diff'] = data.groupby('Pallet_ID')['longitude'].diff()
        data['distance_km'] = np.sqrt(data['lat_diff']**2 + data['lon_diff']**2) * 111  # Approximate km conversion
        
        # Calculate speed and add it to the main DataFrame
        data['speed'] = np.where(data['time_diff'] > 0, 
                                data['distance_km'] / data['time_diff'],
                                0)  # Handle division by zero
        
        # Calculate acceleration and add it to the main DataFrame
        data['acceleration'] = data.groupby('Pallet_ID')['speed'].diff() / data['time_diff']
        
        # Movement patterns
        features['speed'] = data['speed']
        features['acceleration'] = data['acceleration']
        features['avg_speed'] = data.groupby('Pallet_ID')['speed'].transform('mean')
        features['speed_std'] = data.groupby('Pallet_ID')['speed'].transform('std')
        features['speed_variation'] = features['speed_std'] / (features['avg_speed'] + 1e-6)
        
        # Location-based features (without leaking target information)
        features['latitude'] = data['latitude']
        features['longitude'] = data['longitude']
        features['lat_change_rate'] = data.groupby('Pallet_ID')['latitude'].diff() / data['time_diff']
        features['lon_change_rate'] = data.groupby('Pallet_ID')['longitude'].diff() / data['time_diff']
        
        # Time-based patterns
        features['time_since_last_update'] = data['Time_Since_Last_Update'].dt.total_seconds() / 3600
        features['update_frequency'] = data.groupby('Pallet_ID')['timestamp'].transform(
            lambda x: (x.max() - x.min()).total_seconds() / (len(x) * 3600)
        )
        
        # Historical patterns
        features['cumulative_distance'] = data.groupby('Pallet_ID')['distance_km'].transform('cumsum')
        features['distance_per_update'] = features['cumulative_distance'] / data.groupby('Pallet_ID').cumcount().add(1)
        
        # Cluster features if available
        if 'Cluster' in data.columns:
            features['is_clustered'] = (data['Cluster'] != -1).astype(int)
            features['cluster_size'] = data.groupby('Cluster').transform('size')
            features['cluster_size'] = features['cluster_size'].fillna(1)
        else:
            features['is_clustered'] = 0
            features['cluster_size'] = 1
        
        # Handle missing values with more robust strategies
        for col in features.select_dtypes(include=[np.number]).columns:
            if features[col].isnull().any():
                if col in ['speed', 'acceleration', 'lat_change_rate', 'lon_change_rate']:
                    features[col] = features[col].fillna(0)
                elif 'std' in col:
                    features[col] = features[col].fillna(features[col].mean())
                else:
                    features[col] = features[col].fillna(features[col].median())
        
        return features
        
    def train_deviation_predictor(self, historical_data):
        """Train ensemble model to predict route deviations"""
        print("\n🤖 Training Advanced Deviation Prediction Model...")
        
        # Prepare features without using target information
        X = self.prepare_features(historical_data, for_training=True)
        y = (historical_data['Distance_To_Route_KM'] > 5.0).astype(int)
        
        # Print class distribution
        class_dist = y.value_counts(normalize=True) * 100
        print("\n- Class Distribution:")
        print(f"  * On Route: {class_dist[0]:.1f}%")
        print(f"  * Off Route: {class_dist[1]:.1f}%")
        
        # Calculate class weights to handle imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        
        print("\n- Applying class weights to handle imbalance:")
        for cls, weight in class_weight_dict.items():
            print(f"  * Class {cls}: {weight:.2f}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create ensemble model with better handling of imbalance
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight=class_weight_dict,
            random_state=42
        )
        
        svm = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            class_weight=class_weight_dict,
            random_state=42
        )
        
        nn = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        
        # Create voting classifier with adjusted weights
        self.deviation_model = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('svm', svm),
                ('nn', nn)
            ],
            voting='soft',
            weights=[3, 2, 1]  # Give more weight to RF and SVM for imbalanced data
        )
        
        # Train model
        self.deviation_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.deviation_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"- Model Performance:")
        print(f"  * Accuracy: {accuracy*100:.1f}%")
        print(f"  * Precision: {precision*100:.1f}%")
        print(f"  * Recall: {recall*100:.1f}%")
        
        # Feature importance from Random Forest
        rf.fit(X_train_scaled, y_train)
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n- Top deviation indicators:")
        for _, row in importance.head(5).iterrows():
            print(f"  * {row['feature']}: {row['importance']*100:.1f}% importance")
            
    def train_eta_predictor(self, historical_data):
        """Train advanced model to predict delivery times"""
        print("\n⏱️ Training Advanced ETA Prediction Model...")
        
        # Prepare features and ensure distance_km exists
        historical_data = historical_data.copy()
        
        # Calculate distances between consecutive points if not exists
        if 'distance_km' not in historical_data.columns:
            historical_data['lat_diff'] = historical_data.groupby('Pallet_ID')['latitude'].diff()
            historical_data['lon_diff'] = historical_data.groupby('Pallet_ID')['longitude'].diff()
            historical_data['distance_km'] = np.sqrt(
                historical_data['lat_diff']**2 + historical_data['lon_diff']**2
            ) * 111  # Approximate km conversion
            historical_data['distance_km'] = historical_data['distance_km'].fillna(0)
        
        X = self.prepare_features(historical_data)
        
        # Calculate time to destination for each route in a more robust way
        trip_durations = historical_data.groupby('Pallet_ID').agg({
            'timestamp': lambda x: (x.max() - x.min()).total_seconds() / 3600,
            'distance_km': 'sum',
            'Nearest_Route': 'first'
        })
        
        # Calculate average speed per route with better handling of edge cases
        route_speeds = trip_durations.groupby('Nearest_Route').apply(
            lambda x: np.clip(
                x['distance_km'].sum() / (x['timestamp'].sum() + 1e-6),  # km/h
                1,  # Minimum 1 km/h
                120  # Maximum 120 km/h
            )
        )
        
        # Map route speeds back to each pallet
        historical_data['route_avg_speed'] = historical_data['Nearest_Route'].map(route_speeds)
        
        # Calculate remaining distance based on cumulative progress
        historical_data['distance_covered'] = historical_data.groupby('Pallet_ID')['distance_km'].cumsum()
        historical_data['total_route_distance'] = historical_data.groupby('Pallet_ID')['distance_km'].transform('sum')
        historical_data['remaining_distance'] = historical_data['total_route_distance'] - historical_data['distance_covered']
        
        # Calculate expected time to destination with better bounds and handle NaN
        historical_data['time_to_destination'] = np.clip(
            historical_data['remaining_distance'] / (historical_data['route_avg_speed'] + 1e-6),
            0,  # Minimum 0 hours
            72  # Maximum 72 hours
        )
        
        # Handle NaN values in time_to_destination
        historical_data['time_to_destination'] = historical_data['time_to_destination'].fillna(
            historical_data['time_to_destination'].median()
        )
        
        y = historical_data['time_to_destination']
        
        # Print distribution of delivery times
        print("\n- Delivery Time Distribution:")
        print(f"  * Average: {y.mean():.1f} hours")
        print(f"  * Median: {y.median():.1f} hours")
        print(f"  * Min: {y.min():.1f} hours")
        print(f"  * Max: {y.max():.1f} hours")
        
        # Create bins for stratification, handling edge cases
        try:
            strata = pd.qcut(y, q=5, duplicates='drop', labels=False)
        except ValueError:
            # If qcut fails due to too many duplicate values, use simpler bins
            strata = pd.cut(y, bins=5, labels=False)
        
        # Handle any remaining NaN in strata
        strata = strata.fillna(0)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=strata
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train gradient boosting model with better hyperparameters
        self.eta_model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        )
        
        self.eta_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.eta_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate MAPE only for non-zero actual values
        non_zero_mask = y_test > 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        else:
            mape = float('nan')
        
        print(f"\n- ETA Prediction Performance:")
        print(f"  * Mean Absolute Error: {mae:.1f} hours")
        print(f"  * Mean Absolute Percentage Error: {mape:.1f}%")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.eta_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n- Top ETA prediction factors:")
        for _, row in importance.head(5).iterrows():
            print(f"  * {row['feature']}: {row['importance']*100:.1f}% importance")
        
    def predict_deviations(self, current_data):
        """Predict which pallets are at risk of deviation"""
        if self.deviation_model is None:
            raise ValueError("Model not trained yet!")
            
        X = self.prepare_features(current_data)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities of deviation
        deviation_probs = self.deviation_model.predict_proba(X_scaled)[:, 1]
        
        # Create risk categories with confidence levels
        risk_levels = pd.cut(
            deviation_probs,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        return pd.DataFrame({
            'Pallet_ID': current_data['Pallet_ID'],
            'Deviation_Risk': risk_levels,
            'Risk_Probability': deviation_probs,
            'Confidence': np.maximum(deviation_probs, 1 - deviation_probs)
        })
        
    def predict_eta(self, current_data):
        """Predict accurate delivery times with confidence intervals"""
        if self.eta_model is None:
            raise ValueError("Model not trained yet!")
            
        X = self.prepare_features(current_data)
        X_scaled = self.scaler.transform(X)
        
        # Predict hours to destination
        hours_to_destination = self.eta_model.predict(X_scaled)
        
        # Calculate confidence based on prediction variance
        predictions = []
        for i in range(10):  # Use 10 bootstrap samples
            sample_idx = np.random.choice(len(X_scaled), len(X_scaled))
            predictions.append(self.eta_model.predict(X_scaled[sample_idx]))
        predictions = np.array(predictions)
        
        # Calculate confidence as inverse of prediction variance
        confidence_scores = 1 / (1 + np.std(predictions, axis=0))
        
        # Calculate estimated arrival times
        current_time = pd.to_datetime(current_data['timestamp'].iloc[0])
        eta = current_time + pd.to_timedelta(hours_to_destination, unit='h')
        
        return pd.DataFrame({
            'Pallet_ID': current_data['Pallet_ID'],
            'Estimated_Hours_to_Arrival': hours_to_destination,
            'Estimated_Arrival_Time': eta,
            'Prediction_Confidence': confidence_scores
        })
        
    def detect_anomalies(self, current_data):
        """Detect unusual patterns using advanced anomaly detection"""
        features = self.prepare_features(current_data)
        
        # Fit and transform the features
        scaled_features = self.scaler.fit_transform(features)
        
        # Use Isolation Forest with optimized parameters
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(
            n_estimators=200,
            contamination=0.1,
            max_samples='auto',
            random_state=42
        )
        
        # Fit and predict anomalies
        anomaly_scores = iso_forest.fit_predict(scaled_features)
        anomaly_magnitudes = iso_forest.score_samples(scaled_features)
        
        # Calculate confidence scores
        confidence_scores = 1 - np.abs(anomaly_magnitudes - np.mean(anomaly_magnitudes)) / np.std(anomaly_magnitudes)
        
        return pd.DataFrame({
            'Pallet_ID': current_data['Pallet_ID'],
            'Is_Anomaly': anomaly_scores == -1,
            'Anomaly_Score': anomaly_magnitudes,
            'Detection_Confidence': confidence_scores
        }) 