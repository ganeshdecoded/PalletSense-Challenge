import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from haversine import haversine
from datetime import datetime, timedelta
from ..ml.predictive_models import PalletPredictiveSystem

class PalletTracker:
    def __init__(self):
        self.known_routes = None
        self.ongoing_trips = None
        self.risk_thresholds = {
            'distance_km': 5.0,  # Maximum allowed distance from route
            'update_hours': 16.0,  # 2x normal update window
            'revenue_per_km': 10.0  # Example revenue impact per km off route
        }
        self.ml_system = PalletPredictiveSystem()
        
    def load_data(self):
        """Load and prepare the data from CSV files"""
        # Load the CSV files
        self.known_routes = pd.read_csv('data/known_routes.csv')
        self.ongoing_trips = pd.read_csv('data/ongoing_trips.csv')
        
        # Convert timestamps to datetime and ensure they're columns, not indices
        self.known_routes['timestamp'] = pd.to_datetime(self.known_routes['timestamp'])
        self.ongoing_trips['timestamp'] = pd.to_datetime(self.ongoing_trips['timestamp'])
        
        # Reset indices to make sure timestamp is a column
        self.known_routes = self.known_routes.reset_index(drop=True)
        self.ongoing_trips = self.ongoing_trips.reset_index(drop=True)
        
        # Calculate time since last update
        self.ongoing_trips['Time_Since_Last_Update'] = datetime.now() - self.ongoing_trips['timestamp']
        
        print("\n📊 Current Status:")
        print(f"- Monitoring {len(self.ongoing_trips)} active pallets")
        print(f"- Across {len(self.known_routes['Route_ID'].unique())} known routes")
        
        # Step 1: Calculate route distances first
        print("\n📏 Calculating route distances...")
        self._calculate_route_distances(max_distance_km=5.0)
        
        # Step 2: Find clusters
        print("\n🔍 Finding pallet clusters...")
        self.find_traveling_clusters(eps_km=1.0, min_samples=2)
        
        # Step 3: Train ML models
        print("\n🧠 Training AI Models...")
        print("This may take a few minutes for optimal accuracy...")
        self.ml_system.train_deviation_predictor(self.ongoing_trips)
        self.ml_system.train_eta_predictor(self.ongoing_trips)
        
    def _calculate_route_distances(self, max_distance_km=5.0):
        """Calculate basic route distances and adherence"""
        def analyze_pallet(row):
            min_distance = float('inf')
            best_route = None
            
            for route_id in self.known_routes['Route_ID'].unique():
                route_points = self.known_routes[self.known_routes['Route_ID'] == route_id]
                
                for _, route_point in route_points.iterrows():
                    distance = haversine(
                        (row['latitude'], row['longitude']),
                        (route_point['latitude'], route_point['longitude']),
                        unit='km'
                    )
                    if distance < min_distance:
                        min_distance = distance
                        best_route = route_id
            
            # Calculate revenue impact
            revenue_impact = max(0, (min_distance - max_distance_km) * self.risk_thresholds['revenue_per_km'])
            
            return pd.Series({
                'Nearest_Route': best_route,
                'Distance_To_Route_KM': min_distance,
                'Is_On_Route': min_distance <= max_distance_km,
                'Revenue_Impact': revenue_impact
            })
        
        # Calculate distances for each pallet
        results = self.ongoing_trips.apply(analyze_pallet, axis=1)
        self.ongoing_trips = pd.concat([self.ongoing_trips, results], axis=1)
        
    def find_traveling_clusters(self, eps_km=1.0, min_samples=2):
        """Identify groups of pallets traveling together to enhance tracking"""
        def cluster_at_timestamp(group):
            coords = group[['latitude', 'longitude']].values
            clustering = DBSCAN(eps=eps_km/111.12, min_samples=min_samples, metric='euclidean').fit(coords)
            group['Cluster'] = clustering.labels_
            return group
        
        # Reset index to avoid timestamp ambiguity
        self.ongoing_trips = self.ongoing_trips.reset_index(drop=True)
        
        # Group by timestamp and find clusters (using 'h' instead of deprecated 'H')
        grouped = self.ongoing_trips.groupby(pd.to_datetime(self.ongoing_trips['timestamp']).dt.round('h'))
        self.ongoing_trips = grouped.apply(cluster_at_timestamp).reset_index(drop=True)
        
        # Analyze clusters
        clusters = self.ongoing_trips[self.ongoing_trips['Cluster'] != -1]
        cluster_count = len(clusters['Cluster'].unique()) if len(clusters) > 0 else 0
        
        if cluster_count > 0:
            print(f"- Found {cluster_count} groups of pallets traveling together")
            print("- These groups enable:")
            print("  * Enhanced tracking frequency")
            print("  * Early deviation detection")
            print("  * Improved ETA predictions")
            
            # Additional cluster insights
            avg_cluster_size = len(clusters) / cluster_count
            print(f"- Average pallets per group: {avg_cluster_size:.1f}")
        else:
            print("- No pallet groups detected in current data")
            print("- Will use individual pallet analysis")
            
        # Initialize Cluster column if no clusters found
        if 'Cluster' not in self.ongoing_trips.columns:
            self.ongoing_trips['Cluster'] = -1
        
        # Detect anomalies in cluster behavior
        anomalies = self.ml_system.detect_anomalies(self.ongoing_trips)
        anomaly_count = sum(anomalies['Is_Anomaly'])
        if anomaly_count > 0:
            print(f"\n⚠️ Detected {anomaly_count} pallets with unusual movement patterns")
        
    def calculate_route_adherence(self, max_distance_km=5.0):
        """Analyze route adherence with ML insights"""
        # Update route distances
        self._calculate_route_distances(max_distance_km)
        
        # Get ML predictions
        deviation_risks = self.ml_system.predict_deviations(self.ongoing_trips)
        eta_predictions = self.ml_system.predict_eta(self.ongoing_trips)
        
        # Calculate business metrics
        total_pallets = len(self.ongoing_trips)
        off_route = sum(~self.ongoing_trips['Is_On_Route'])
        delayed_updates = sum(
            self.ongoing_trips['Time_Since_Last_Update'].dt.total_seconds() / 3600 > 
            self.risk_thresholds['update_hours']
        )
        total_revenue_impact = self.ongoing_trips['Revenue_Impact'].sum()
        
        print("\n🚨 Risk Analysis:")
        print(f"- {off_route} pallets ({off_route/total_pallets*100:.1f}%) are off their expected routes")
        print(f"- {delayed_updates} pallets have delayed updates")
        print(f"- Potential daily revenue impact: ${total_revenue_impact:.2f}")
        
        # Show ML insights
        high_risk = sum(deviation_risks['Deviation_Risk'].isin(['High', 'Very High']))
        print(f"\n🤖 AI Predictions:")
        print(f"- {high_risk} pallets at high risk of future deviation")
        print("- Average predicted arrival times:")
        for route in self.known_routes['Route_ID'].unique():
            route_etas = eta_predictions[
                self.ongoing_trips['Nearest_Route'] == route
            ]['Estimated_Hours_to_Arrival']
            if len(route_etas) > 0:
                print(f"  * Route {route}: {route_etas.mean():.1f} hours")
        
        # Calculate route-specific insights
        route_metrics = self.ongoing_trips.groupby('Nearest_Route').agg({
            'Is_On_Route': 'mean',
            'Revenue_Impact': 'sum'
        }).sort_values('Revenue_Impact', ascending=False)
        
        print("\n📍 Route-Specific Insights:")
        for route_id, row in route_metrics.head(3).iterrows():
            print(f"- Route {route_id}:")
            print(f"  * {row['Is_On_Route']*100:.1f}% adherence rate")
            print(f"  * ${row['Revenue_Impact']:.2f} potential daily losses")
            
            # Add ML insights for this route
            route_risks = deviation_risks[self.ongoing_trips['Nearest_Route'] == route_id]
            high_risk_count = sum(route_risks['Deviation_Risk'].isin(['High', 'Very High']))
            if high_risk_count > 0:
                print(f"  * {high_risk_count} pallets at high risk of deviation")
        
    def detect_stationary_pallets(self, time_threshold_hours=24):
        """Identify potentially lost or abandoned pallets"""
        def is_stationary(group):
            if len(group) < 2:
                return False
            
            first_point = group.iloc[0]
            max_distance_km = 0.5
            
            for _, point in group.iterrows():
                distance = haversine(
                    (first_point['latitude'], first_point['longitude']),
                    (point['latitude'], point['longitude']),
                    unit='km'
                )
                if distance > max_distance_km:
                    return False
            return True
        
        # Find stationary pallets
        stationary = self.ongoing_trips.groupby('Pallet_ID').apply(is_stationary)
        stationary_pallets = stationary[stationary].index.tolist()
        
        # Get anomaly detection results for stationary pallets
        if len(stationary_pallets) > 0:
            stationary_data = self.ongoing_trips[
                self.ongoing_trips['Pallet_ID'].isin(stationary_pallets)
            ]
            anomalies = self.ml_system.detect_anomalies(stationary_data)
            suspicious_stops = sum(anomalies['Is_Anomaly'])
        else:
            suspicious_stops = 0
        
        print("\n⚠️ Stationary Pallet Detection:")
        if len(stationary_pallets) > 0:
            print(f"- Found {len(stationary_pallets)} potentially lost/abandoned pallets")
            if suspicious_stops > 0:
                print(f"- {suspicious_stops} stops flagged as suspicious by AI")
            print("- Recommended actions:")
            print("  * Immediate location verification")
            print("  * Customer contact for status update")
            print("  * Billing review for extended storage")
        else:
            print("- No stationary pallets detected in current data")
            print("- Continue monitoring for extended stays")
        
        return stationary_pallets

if __name__ == "__main__":
    # Initialize the tracker
    tracker = PalletTracker()
    
    # Load the data
    tracker.load_data()
    
    # Calculate route adherence
    tracker.calculate_route_adherence()
    
    # Detect stationary pallets
    stationary_pallets = tracker.detect_stationary_pallets() 