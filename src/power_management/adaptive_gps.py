from datetime import datetime, timedelta
import numpy as np

class AdaptiveGPSController:
    def __init__(self):
        self.base_ping_interval_hours = 8  # Base interval between pings
        self.min_ping_interval_hours = 4   # Minimum interval between pings
        self.max_ping_interval_hours = 12  # Maximum interval between pings
        
        # Risk factors and their weights
        self.risk_factors = {
            'near_known_location': -0.3,   # Reduce frequency near known points
            'high_risk_area': 0.4,         # Increase frequency in risky areas
            'in_cluster': -0.2,            # Reduce if traveling with others
            'historical_deviation': 0.3,    # Increase if route often has issues
            'mesh_density': -0.2,          # Reduce if many nearby nodes
            'battery_level': -0.1,         # Reduce if battery is low
            'speed': 0.2                   # Increase if moving fast
        }
        
    def calculate_ping_interval(self, context):
        """Calculate optimal ping interval based on context"""
        # Start with base interval
        adjustment = 0
        
        # Apply risk factors
        if context.get('near_known_location', False):
            adjustment += self.risk_factors['near_known_location']
            
        if context.get('high_risk_area', False):
            adjustment += self.risk_factors['high_risk_area']
            
        if context.get('in_cluster', False):
            adjustment += self.risk_factors['in_cluster']
            
        if context.get('historical_deviation', 0) > 0.5:
            adjustment += self.risk_factors['historical_deviation']
            
        mesh_density = context.get('mesh_density', 0)
        if mesh_density > 0.1:  # More than 0.1 points per square km
            adjustment += self.risk_factors['mesh_density']
            
        battery_level = context.get('battery_level', 1.0)
        if battery_level < 0.3:  # Below 30% battery
            adjustment += self.risk_factors['battery_level']
            
        speed = context.get('speed', 0)
        if speed > 60:  # Moving fast (km/h)
            adjustment += self.risk_factors['speed']
            
        # Calculate new interval
        new_interval = self.base_ping_interval_hours * (1 - adjustment)
        
        # Ensure within bounds
        new_interval = max(self.min_ping_interval_hours,
                         min(self.max_ping_interval_hours, new_interval))
                         
        return new_interval
        
    def should_ping_now(self, last_ping_time, context):
        """Determine if a ping should be made now"""
        # Calculate optimal interval
        optimal_interval = self.calculate_ping_interval(context)
        
        # Check if enough time has passed
        time_since_last_ping = (datetime.now() - last_ping_time).total_seconds() / 3600
        
        # Always ping if we're beyond max interval
        if time_since_last_ping >= self.max_ping_interval_hours:
            return True
            
        # Check if we've reached optimal interval
        return time_since_last_ping >= optimal_interval
        
    def estimate_battery_impact(self, ping_frequency):
        """Estimate battery impact of a given ping frequency"""
        # Assume each ping uses 0.1% battery
        pings_per_day = 24 / ping_frequency
        daily_battery_usage = pings_per_day * 0.1
        
        return {
            'pings_per_day': pings_per_day,
            'daily_battery_usage': daily_battery_usage,
            'estimated_battery_life_days': 100 / daily_battery_usage  # Assuming 100% battery
        }
        
    def optimize_for_route(self, route_points, known_locations):
        """Optimize ping strategy for a specific route"""
        optimized_points = []
        
        for i, point in enumerate(route_points):
            # Calculate distance to nearest known location
            min_distance = float('inf')
            for loc in known_locations:
                distance = np.sqrt(
                    (point['latitude'] - loc['latitude'])**2 +
                    (point['longitude'] - loc['longitude'])**2
                )
                min_distance = min(min_distance, distance)
            
            # Create context for this point
            context = {
                'near_known_location': min_distance < 0.1,  # Within 0.1 degrees
                'high_risk_area': False,  # Could be determined by historical data
                'in_cluster': False,      # Would be determined by real-time data
                'historical_deviation': 0, # Would come from historical data
                'mesh_density': 0,        # Would come from mesh network
                'battery_level': 1.0,     # Would come from device
                'speed': 0                # Would be calculated from movement
            }
            
            # Calculate optimal interval
            optimal_interval = self.calculate_ping_interval(context)
            
            optimized_points.append({
                **point,
                'recommended_interval': optimal_interval,
                'context': context
            })
            
        return optimized_points 