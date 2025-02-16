import numpy as np
from haversine import haversine
from datetime import datetime, timedelta

class MeshNetworkTracker:
    def __init__(self):
        self.network_nodes = {
            'fixed_points': [],  # warehouses, stores, known locations
            'mobile_points': []  # pallets, trucks
        }
        self.max_distance_km = 50  # Maximum distance to consider for mesh network
        
    def add_fixed_point(self, point_id, latitude, longitude, point_type='warehouse'):
        """Add a fixed point (warehouse, store, etc.) to the mesh network"""
        self.network_nodes['fixed_points'].append({
            'id': point_id,
            'latitude': latitude,
            'longitude': longitude,
            'type': point_type
        })
        
    def update_mobile_point(self, point_id, latitude, longitude, timestamp, point_type='pallet'):
        """Update or add a mobile point (pallet, truck) in the mesh network"""
        # Remove old position if exists
        self.network_nodes['mobile_points'] = [
            p for p in self.network_nodes['mobile_points'] if p['id'] != point_id
        ]
        
        # Add new position
        self.network_nodes['mobile_points'].append({
            'id': point_id,
            'latitude': latitude,
            'longitude': longitude,
            'timestamp': timestamp,
            'type': point_type
        })
        
    def find_nearby_nodes(self, latitude, longitude, max_age_hours=2):
        """Find all nodes (fixed and mobile) near a given position"""
        nearby_nodes = {
            'fixed_points': [],
            'mobile_points': []
        }
        
        # Check fixed points
        for point in self.network_nodes['fixed_points']:
            distance = haversine(
                (latitude, longitude),
                (point['latitude'], point['longitude']),
                unit='km'
            )
            if distance <= self.max_distance_km:
                nearby_nodes['fixed_points'].append({
                    **point,
                    'distance': distance
                })
                
        # Check mobile points
        current_time = datetime.now()
        for point in self.network_nodes['mobile_points']:
            # Skip old points
            point_time = point['timestamp']
            if isinstance(point_time, str):
                point_time = datetime.fromisoformat(point_time.replace('Z', '+00:00'))
            if (current_time - point_time) > timedelta(hours=max_age_hours):
                continue
                
            distance = haversine(
                (latitude, longitude),
                (point['latitude'], point['longitude']),
                unit='km'
            )
            if distance <= self.max_distance_km:
                nearby_nodes['mobile_points'].append({
                    **point,
                    'distance': distance
                })
                
        return nearby_nodes
    
    def estimate_position(self, pallet_id, timestamp):
        """Estimate position of a pallet using mesh network data"""
        # Find the pallet's last known position
        pallet_point = next(
            (p for p in self.network_nodes['mobile_points'] if p['id'] == pallet_id),
            None
        )
        
        if not pallet_point:
            return None
            
        # Find nearby nodes
        nearby = self.find_nearby_nodes(
            pallet_point['latitude'],
            pallet_point['longitude']
        )
        
        # If we have enough nearby points, use them to estimate position
        total_points = len(nearby['fixed_points']) + len(nearby['mobile_points'])
        if total_points < 2:
            return None
            
        # Calculate weighted average position based on distance
        weights = []
        positions = []
        
        # Add fixed points (higher weight)
        for point in nearby['fixed_points']:
            positions.append([point['latitude'], point['longitude']])
            weights.append(1.5 * (self.max_distance_km - point['distance']))
            
        # Add mobile points (lower weight)
        for point in nearby['mobile_points']:
            if point['id'] != pallet_id:  # Don't use the pallet's own position
                positions.append([point['latitude'], point['longitude']])
                weights.append(self.max_distance_km - point['distance'])
                
        # Calculate weighted average
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        positions = np.array(positions)
        
        estimated_position = np.average(positions, weights=weights, axis=0)
        
        return {
            'latitude': float(estimated_position[0]),
            'longitude': float(estimated_position[1]),
            'confidence': min(1.0, total_points / 5),  # Confidence based on number of nearby points
            'nearby_fixed': len(nearby['fixed_points']),
            'nearby_mobile': len(nearby['mobile_points']) - 1  # Subtract self
        }
    
    def calculate_mesh_density(self, latitude, longitude):
        """Calculate the density of the mesh network at a given position"""
        nearby = self.find_nearby_nodes(latitude, longitude)
        total_points = len(nearby['fixed_points']) + len(nearby['mobile_points'])
        return total_points / (np.pi * self.max_distance_km ** 2)  # points per square km 