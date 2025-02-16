import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
from collections import defaultdict

class PalletMonitor:
    def __init__(self):
        self.stoppage_thresholds = {
            'warning': 12,     # hours
            'critical': 24,    # hours
            'loss_risk': 48    # hours
        }
        self.historical_loss_points = defaultdict(int)
        
    def analyze_stoppages(self, pallet_data):
        """Analyze pallet stoppages and identify potential losses"""
        results = []
        
        # Group by pallet to analyze movement
        for pallet_id, group in pallet_data.groupby('Pallet_ID'):
            group = group.sort_values('timestamp')
            
            # Calculate time differences and distances
            time_diffs = group['timestamp'].diff()
            distances = np.sqrt(
                group['latitude'].diff()**2 + 
                group['longitude'].diff()**2
            ) * 111  # Approximate km conversion
            
            # Identify stationary periods
            is_stationary = (distances < 0.1)  # Less than 100m movement
            stationary_duration = time_diffs[is_stationary].sum()
            
            if stationary_duration is pd.NaT:
                continue
                
            hours_stationary = stationary_duration.total_seconds() / 3600
            
            # Get last known location
            last_location = group.iloc[-1]
            
            # Update historical loss points
            location_key = f"{last_location['latitude']:.3f},{last_location['longitude']:.3f}"
            if hours_stationary >= self.stoppage_thresholds['critical']:
                self.historical_loss_points[location_key] += 1
            
            # Determine risk level
            if hours_stationary >= self.stoppage_thresholds['loss_risk']:
                risk_level = 'HIGH_RISK'
                alert_level = '🔴'
            elif hours_stationary >= self.stoppage_thresholds['critical']:
                risk_level = 'CRITICAL'
                alert_level = '🟠'
            elif hours_stationary >= self.stoppage_thresholds['warning']:
                risk_level = 'WARNING'
                alert_level = '🟡'
            else:
                risk_level = 'NORMAL'
                alert_level = '🟢'
            
            results.append({
                'Pallet_ID': pallet_id,
                'Hours_Stationary': hours_stationary,
                'Risk_Level': risk_level,
                'Alert_Level': alert_level,
                'Last_Latitude': last_location['latitude'],
                'Last_Longitude': last_location['longitude'],
                'Last_Update': last_location['timestamp'],
                'Historical_Loss_Risk': self.historical_loss_points[location_key]
            })
        
        return pd.DataFrame(results)
    
    def identify_loss_hotspots(self, min_incidents=2):
        """Identify locations with multiple loss incidents"""
        hotspots = []
        
        for location, count in self.historical_loss_points.items():
            if count >= min_incidents:
                lat, lon = map(float, location.split(','))
                hotspots.append({
                    'latitude': lat,
                    'longitude': lon,
                    'incident_count': count
                })
        
        return pd.DataFrame(hotspots)
    
    def track_pallet_groups(self, pallet_data, max_distance_km=1.0, min_group_size=2):
        """Track groups of pallets moving together"""
        groups = []
        
        # Analyze each timestamp
        for timestamp, frame in pallet_data.groupby('timestamp'):
            # Prepare coordinates for clustering
            coords = frame[['latitude', 'longitude']].values
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(
                eps=max_distance_km/111.12,  # Convert km to degrees
                min_samples=min_group_size,
                metric='euclidean'
            ).fit(coords)
            
            # Get unique clusters (excluding noise points marked as -1)
            unique_clusters = set(clustering.labels_[clustering.labels_ != -1])
            
            for cluster_id in unique_clusters:
                cluster_mask = clustering.labels_ == cluster_id
                cluster_pallets = frame[cluster_mask]
                
                # Calculate cluster statistics
                center_lat = cluster_pallets['latitude'].mean()
                center_lon = cluster_pallets['longitude'].mean()
                
                groups.append({
                    'Timestamp': timestamp,
                    'Group_ID': f"G{len(groups)+1}",
                    'Pallets': cluster_pallets['Pallet_ID'].tolist(),
                    'Group_Size': sum(cluster_mask),
                    'Center_Latitude': center_lat,
                    'Center_Longitude': center_lon,
                    'Spread_KM': np.max([
                        np.sqrt(
                            (row['latitude'] - center_lat)**2 + 
                            (row['longitude'] - center_lon)**2
                        ) * 111
                        for _, row in cluster_pallets.iterrows()
                    ])
                })
        
        return pd.DataFrame(groups)
    
    def generate_monitoring_alerts(self, stoppage_analysis, group_tracking):
        """Generate comprehensive monitoring alerts"""
        alerts = []
        
        # Process stoppage alerts
        for _, row in stoppage_analysis.iterrows():
            if row['Risk_Level'] != 'NORMAL':
                alert = f"{row['Alert_Level']} Stoppage Alert: Pallet {row['Pallet_ID']} "
                
                if row['Risk_Level'] == 'HIGH_RISK':
                    alert += f"CRITICAL: Stationary for {row['Hours_Stationary']:.1f} hours! "
                    if row['Historical_Loss_Risk'] > 1:
                        alert += f"Location has {row['Historical_Loss_Risk']} previous incidents. "
                    alert += "Immediate investigation required!"
                elif row['Risk_Level'] == 'CRITICAL':
                    alert += f"has been stationary for {row['Hours_Stationary']:.1f} hours. "
                    alert += "Requires attention."
                else:
                    alert += f"stopped for {row['Hours_Stationary']:.1f} hours. Monitor situation."
                
                alerts.append({
                    'Type': 'STOPPAGE',
                    'Alert_Text': alert,
                    'Severity': row['Risk_Level'],
                    'Timestamp': row['Last_Update']
                })
        
        # Process group tracking alerts
        if not group_tracking.empty:
            latest_groups = group_tracking.sort_values('Timestamp').groupby('Group_ID').last()
            
            for _, group in latest_groups.iterrows():
                alert = f"📦 Group Tracking: {group['Group_Size']} pallets traveling together "
                alert += f"(Group {group['Group_ID']}). "
                alert += f"Spread: {group['Spread_KM']:.1f}km"
                
                alerts.append({
                    'Type': 'GROUP',
                    'Alert_Text': alert,
                    'Severity': 'INFO',
                    'Timestamp': group['Timestamp']
                })
        
        return pd.DataFrame(alerts) 