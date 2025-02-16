class PalletMonitor:
    def analyze_stoppages(self, tracking_data, min_stationary_minutes=30):
        """Analyze pallet stoppages and calculate stationary durations"""
        print("- Analyzing pallet stoppages...")
        
        # Convert timestamps to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(tracking_data['timestamp']):
            tracking_data['timestamp'] = pd.to_datetime(tracking_data['timestamp'])
            
        # Sort data by pallet and time
        tracking_data = tracking_data.sort_values(['Pallet_ID', 'timestamp'])
        
        # Calculate time differences and distances between consecutive points
        tracking_data['time_diff'] = tracking_data.groupby('Pallet_ID')['timestamp'].diff()
        tracking_data['time_diff'] = tracking_data['time_diff'].dt.total_seconds() / 60  # Convert to minutes
        
        # Calculate distances using haversine formula
        tracking_data['distance'] = tracking_data.groupby('Pallet_ID').apply(
            lambda x: pd.Series(
                [0] + [
                    haversine(
                        (x['latitude'].iloc[i], x['longitude'].iloc[i]),
                        (x['latitude'].iloc[i+1], x['longitude'].iloc[i+1]),
                        unit='km'
                    ) 
                    for i in range(len(x)-1)
                ]
            )
        ).values
        
        # Identify stationary periods
        tracking_data['is_stationary'] = (
            (tracking_data['distance'] < 0.1) &  # Less than 100m movement
            (tracking_data['time_diff'] >= min_stationary_minutes)
        )
        
        # Calculate stationary durations
        stationary_periods = []
        for pallet_id, group in tracking_data.groupby('Pallet_ID'):
            stationary_mask = group['is_stationary']
            if not stationary_mask.any():
                continue
                
            # Find continuous stationary periods
            stationary_groups = (stationary_mask != stationary_mask.shift()).cumsum()[stationary_mask]
            for _, period in group[stationary_mask].groupby(stationary_groups):
                stationary_periods.append({
                    'Pallet_ID': pallet_id,
                    'start_time': period['timestamp'].iloc[0],
                    'end_time': period['timestamp'].iloc[-1],
                    'duration_minutes': (period['timestamp'].iloc[-1] - period['timestamp'].iloc[0]).total_seconds() / 60,
                    'latitude': period['latitude'].mean(),
                    'longitude': period['longitude'].mean(),
                    'location': period['location'].iloc[0] if 'location' in period.columns else 'Unknown'
                })
        
        self.stationary_periods = pd.DataFrame(stationary_periods)
        print(f"- Found {len(self.stationary_periods)} stationary periods")
        return self.stationary_periods

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

    def detect_anomalies(self, tracking_data, window_size=5):
        """
        Enhanced anomaly detection using statistical methods and rolling windows
        """
        results = pd.DataFrame()
        
        # Group by pallet
        for pallet_id, pallet_data in tracking_data.groupby('Pallet_ID'):
            # Sort by timestamp
            pallet_data = pallet_data.sort_values('timestamp')
            
            # Calculate speed and acceleration
            time_diffs = pallet_data['timestamp'].diff().dt.total_seconds() / 3600  # hours
            distances = self._calculate_distances(pallet_data)
            speeds = distances / time_diffs.fillna(1)
            accelerations = speeds.diff() / time_diffs.fillna(1)
            
            # Calculate rolling statistics
            speed_mean = speeds.rolling(window=window_size, min_periods=1).mean()
            speed_std = speeds.rolling(window=window_size, min_periods=1).std()
            acc_mean = accelerations.rolling(window=window_size, min_periods=1).mean()
            acc_std = accelerations.rolling(window=window_size, min_periods=1).std()
            
            # Z-scores for speed and acceleration
            speed_zscore = np.abs((speeds - speed_mean) / speed_std.fillna(1))
            acc_zscore = np.abs((accelerations - acc_mean) / acc_std.fillna(1))
            
            # Calculate route deviation
            route_distances = pallet_data['Distance_To_Route_KM']
            route_mean = route_distances.rolling(window=window_size, min_periods=1).mean()
            route_std = route_distances.rolling(window=window_size, min_periods=1).std()
            route_zscore = np.abs((route_distances - route_mean) / route_std.fillna(1))
            
            # Time gap analysis
            time_gap_zscore = np.abs((time_diffs - time_diffs.mean()) / time_diffs.std())
            
            # Combined anomaly score using weighted factors
            anomaly_scores = (
                0.3 * speed_zscore +      # Speed anomalies
                0.2 * acc_zscore +        # Acceleration anomalies
                0.3 * route_zscore +      # Route deviation anomalies
                0.2 * time_gap_zscore     # Time gap anomalies
            )
            
            # Dynamic thresholding based on historical patterns
            threshold = np.percentile(anomaly_scores[~np.isnan(anomaly_scores)], 95)
            
            # Detect anomalies with additional context
            anomalies = pd.DataFrame({
                'Pallet_ID': pallet_id,
                'timestamp': pallet_data['timestamp'],
                'Anomaly_Score': anomaly_scores,
                'Speed_Score': speed_zscore,
                'Acceleration_Score': acc_zscore,
                'Route_Score': route_zscore,
                'Time_Gap_Score': time_gap_zscore,
                'Is_Anomaly': anomaly_scores > threshold,
                'Anomaly_Type': self._classify_anomaly_type(
                    speed_zscore, acc_zscore, route_zscore, time_gap_zscore
                )
            })
            
            results = pd.concat([results, anomalies])
        
        return results
    
    def _calculate_distances(self, data):
        """Calculate distances between consecutive points using Haversine formula"""
        lat1 = data['latitude'].shift()
        lon1 = data['longitude'].shift()
        lat2 = data['latitude']
        lon2 = data['longitude']
        
        R = 6371  # Earth's radius in kilometers
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _classify_anomaly_type(self, speed_score, acc_score, route_score, time_score):
        """Classify the type of anomaly based on component scores"""
        max_scores = pd.DataFrame({
            'Speed': speed_score,
            'Acceleration': acc_score,
            'Route': route_score,
            'Time_Gap': time_score
        })
        
        anomaly_types = []
        for _, scores in max_scores.iterrows():
            if scores.max() <= 2:  # No significant anomaly
                anomaly_types.append('Normal')
            else:
                primary_factor = scores.idxmax()
                if primary_factor == 'Speed':
                    anomaly_types.append('Unusual Speed')
                elif primary_factor == 'Acceleration':
                    anomaly_types.append('Sudden Movement')
                elif primary_factor == 'Route':
                    anomaly_types.append('Route Deviation')
                else:
                    anomaly_types.append('Irregular Updates')
        
        return anomaly_types 