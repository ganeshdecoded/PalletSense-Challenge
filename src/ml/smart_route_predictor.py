import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from haversine import haversine
from sklearn.preprocessing import StandardScaler

class SmartRoutePredictor:
    def __init__(self):
        self.location_predictor = RandomForestRegressor(
            n_estimators=1000,  # Increased from 200
            max_depth=20,       # Increased from 10
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        self.deviation_thresholds = {
            'minor': 1.0,    # km
            'medium': 3.0,   # km
            'critical': 5.0  # km
        }
        self.scaler = None  # Will be initialized during training
        
    def train_location_predictor(self, historical_data):
        """Train model to predict next location based on historical patterns"""
        print("- Preparing location prediction features...")
        features = self._prepare_location_features(historical_data)
        
        # Prepare target variables (next locations)
        next_locations = historical_data.groupby('Pallet_ID')[['latitude', 'longitude']].shift(-1)
        next_locations = next_locations.ffill()  # Use ffill instead of deprecated fillna(method='ffill')
        
        # Remove rows where we don't have next locations
        valid_mask = ~next_locations.isna().any(axis=1)
        X = features[valid_mask.values]  # Use .values to avoid reindexing warning
        y = next_locations[valid_mask.values]
        
        print(f"- Training on {len(X)} location points...")
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Improve model parameters for better accuracy
        self.location_predictor = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Train the model
        self.location_predictor.fit(X_scaled, y)
        
        # Calculate prediction accuracy
        predictions = self.location_predictor.predict(X_scaled)
        mean_distance_error = np.mean([
            haversine(
                (y.iloc[i]['latitude'], y.iloc[i]['longitude']),
                (predictions[i][0], predictions[i][1]),
                unit='km'
            )
            for i in range(len(predictions))
        ])
        
        print(f"- Average prediction error: {mean_distance_error:.2f}km")
        
        # Print feature importance
        importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.location_predictor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n- Top location prediction factors:")
        for _, row in importance.head(5).iterrows():
            print(f"  * {row['feature']}: {row['importance']*100:.1f}% importance")
        
    def _prepare_location_features(self, data):
        """Prepare features for location prediction"""
        data = data.copy().sort_values(['Pallet_ID', 'timestamp'])
        features = pd.DataFrame()
        
        # Current location
        features['latitude'] = data['latitude']
        features['longitude'] = data['longitude']
        
        # Enhanced time features with cyclical encoding
        timestamps = pd.to_datetime(data['timestamp'])
        features['hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
        features['month_sin'] = np.sin(2 * np.pi * timestamps.dt.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * timestamps.dt.month / 12)
        
        # Calculate time differences
        time_diffs = data.groupby('Pallet_ID')['timestamp'].diff().dt.total_seconds() / 3600  # hours
        
        # Calculate distances between consecutive points
        lat_diffs = data.groupby('Pallet_ID')['latitude'].diff()
        lon_diffs = data.groupby('Pallet_ID')['longitude'].diff()
        distances = np.sqrt(lat_diffs**2 + lon_diffs**2) * 111  # km
        
        # Enhanced speed calculations
        features['speed'] = np.where(time_diffs > 0, 
                                   distances / time_diffs,
                                   0)  # Handle division by zero
        features['speed_ma'] = features.groupby(data['Pallet_ID'])['speed'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Enhanced direction features
        features['direction'] = np.arctan2(lat_diffs, lon_diffs)
        features['direction_change'] = features.groupby(data['Pallet_ID'])['direction'].diff()
        features['direction_ma'] = features.groupby(data['Pallet_ID'])['direction'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Historical patterns with more sophisticated aggregations
        for window in [3, 5, 10]:
            features[f'avg_speed_{window}'] = features.groupby(data['Pallet_ID'])['speed'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            features[f'speed_std_{window}'] = features.groupby(data['Pallet_ID'])['speed'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            features[f'direction_std_{window}'] = features.groupby(data['Pallet_ID'])['direction'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Additional movement features
        features['acceleration'] = features.groupby(data['Pallet_ID'])['speed'].diff() / time_diffs.fillna(1)
        features['jerk'] = features.groupby(data['Pallet_ID'])['acceleration'].diff() / time_diffs.fillna(1)
        
        # Cumulative distance and time features
        features['cumulative_distance'] = features.groupby(data['Pallet_ID'])['speed'].cumsum() * time_diffs.fillna(0)
        features['trip_progress'] = features['cumulative_distance'] / features.groupby(data['Pallet_ID'])['cumulative_distance'].transform('max')
        
        # Handle any remaining NaN values with more sophisticated strategies
        for col in features.select_dtypes(include=[np.number]).columns:
            if features[col].isnull().any():
                if 'direction' in col:
                    features[col] = features[col].fillna(0)
                elif 'std' in col:
                    features[col] = features[col].fillna(features[col].mean())
                else:
                    features[col] = features[col].fillna(features[col].median())
        
        return features
        
    def predict_next_location(self, current_data):
        """Predict next location for pallets with missing GPS signals"""
        features = self._prepare_location_features(current_data)
        predictions = self.location_predictor.predict(features)
        
        return pd.DataFrame({
            'Pallet_ID': current_data['Pallet_ID'],
            'Predicted_Latitude': predictions[:, 0],
            'Predicted_Longitude': predictions[:, 1],
            'Prediction_Time': current_data['timestamp'] + pd.Timedelta(hours=1)
        })
        
    def analyze_deviation(self, pallet_data, route_data):
        """Analyze deviation severity and provide smart alerts"""
        results = []
        
        for _, pallet in pallet_data.iterrows():
            # Find nearest route point
            min_distance = float('inf')
            nearest_point = None
            
            route_points = route_data[route_data['Route_ID'] == pallet['Nearest_Route']]
            for _, route_point in route_points.iterrows():
                distance = haversine(
                    (pallet['latitude'], pallet['longitude']),
                    (route_point['latitude'], route_point['longitude']),
                    unit='km'
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = route_point
            
            # Determine deviation severity
            if min_distance <= self.deviation_thresholds['minor']:
                severity = 'ON_ROUTE'
            elif min_distance <= self.deviation_thresholds['medium']:
                severity = 'MINOR_DEVIATION'
            elif min_distance <= self.deviation_thresholds['critical']:
                severity = 'MEDIUM_DEVIATION'
            else:
                severity = 'CRITICAL_DEVIATION'
            
            # Calculate direction relative to route
            if nearest_point is not None:
                direction_to_route = np.arctan2(
                    nearest_point['latitude'] - pallet['latitude'],
                    nearest_point['longitude'] - pallet['longitude']
                )
                
                # Estimate time to return to route based on average speed
                if 'speed' in pallet:
                    time_to_return = min_distance / (max(pallet['speed'], 1) + 1e-6)
                else:
                    time_to_return = None
            else:
                direction_to_route = None
                time_to_return = None
            
            results.append({
                'Pallet_ID': pallet['Pallet_ID'],
                'Deviation_Distance': min_distance,
                'Severity': severity,
                'Direction_To_Route': direction_to_route,
                'Estimated_Return_Time': time_to_return,
                'Timestamp': pallet['timestamp']
            })
        
        return pd.DataFrame(results) 