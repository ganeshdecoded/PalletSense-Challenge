import folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from plotly.subplots import make_subplots

class PalletVisualizer:
    def __init__(self, tracker):
        self.tracker = tracker
        
    def create_interactive_map(self, center_lat=39.8283, center_lon=-98.5795):
        """Create an interactive map showing pallet locations and predictions"""
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        # Add known routes with better styling
        for route_id in self.tracker.known_routes['Route_ID'].unique():
            route_data = self.tracker.known_routes[self.tracker.known_routes['Route_ID'] == route_id]
            points = route_data[['latitude', 'longitude']].values.tolist()
            
            folium.PolyLine(
                points,
                weight=3,
                color='blue',
                opacity=0.6,
                popup=f'Route {route_id}',
                tooltip=f'Route {route_id}'
            ).add_to(m)
        
        # Get ML predictions
        deviation_risks = self.tracker.ml_system.predict_deviations(self.tracker.ongoing_trips)
        eta_predictions = self.tracker.ml_system.predict_eta(self.tracker.ongoing_trips)
        anomalies = self.tracker.ml_system.detect_anomalies(self.tracker.ongoing_trips)
        
        # Create legend HTML
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <h4>Pallet Status</h4>
        <p><span style="color: green;">●</span> On Track</p>
        <p><span style="color: yellow;">●</span> Minor Deviation</p>
        <p><span style="color: orange;">●</span> Significant Deviation</p>
        <p><span style="color: red;">●</span> Critical Deviation</p>
        <p><span style="color: purple;">●</span> Anomalous Behavior</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add pallet locations with enhanced risk indicators
        for idx, row in self.tracker.ongoing_trips.iterrows():
            # Get risk level
            risk_level = deviation_risks.loc[
                deviation_risks['Pallet_ID'] == row['Pallet_ID'],
                'Deviation_Risk'
            ].iloc[0]
            
            is_anomaly = anomalies.loc[
                anomalies['Pallet_ID'] == row['Pallet_ID'],
                'Is_Anomaly'
            ].iloc[0]
            
            eta = eta_predictions.loc[
                eta_predictions['Pallet_ID'] == row['Pallet_ID'],
                'Estimated_Arrival_Time'
            ].iloc[0]
            
            # Determine color and status based on risk and anomaly
            if is_anomaly:
                color = 'purple'
                status = '⚠️ ANOMALY DETECTED'
                radius = 10
            else:
                if row['Distance_To_Route_KM'] <= 1.0:  # Within 1km
                    color = 'green'
                    status = '✅ ON TRACK'
                    radius = 8
                elif row['Distance_To_Route_KM'] <= 3.0:  # Within 3km
                    color = 'yellow'
                    status = '⚠️ MINOR DEVIATION'
                    radius = 8
                elif row['Distance_To_Route_KM'] <= 5.0:  # Within 5km
                    color = 'orange'
                    status = '🚨 SIGNIFICANT DEVIATION'
                    radius = 9
                else:
                    color = 'red'
                    status = '🔴 CRITICAL DEVIATION'
                    radius = 10
            
            # Create detailed popup content
            popup_html = f"""
            <div style="min-width: 200px;">
                <h4>Pallet {row['Pallet_ID']}</h4>
                <p><b>Status:</b> {status}</p>
                <p><b>Last Update:</b> {row['timestamp']}</p>
                <p><b>Hours Since Update:</b> {row['Time_Since_Last_Update'].total_seconds() / 3600:.1f}</p>
                <p><b>Distance from Route:</b> {row['Distance_To_Route_KM']:.2f} km</p>
                <p><b>Estimated Arrival:</b> {eta}</p>
                <p><b>Revenue Impact:</b> ${row['Revenue_Impact']:.2f}</p>
            </div>
            """
            
            # Add circle marker with enhanced styling
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=color,
                fill=True,
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Pallet {row['Pallet_ID']} - {status}"
            ).add_to(m)
            
            # If anomaly or critical deviation, add attention circle
            if is_anomaly or color == 'red':
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=radius + 5,
                    color=color,
                    fill=False,
                    weight=1,
                    opacity=0.5
                ).add_to(m)
        
        return m
    
    def create_dashboard(self):
        """Create AI-powered dashboard"""
        fig1 = self._create_risk_prediction()
        fig2 = self._create_eta_analysis()
        fig3 = self._create_anomaly_detection()
        
        return [fig1, fig2, fig3]
    
    def _create_risk_prediction(self):
        """Create risk prediction visualization"""
        deviation_risks = self.tracker.ml_system.predict_deviations(self.tracker.ongoing_trips)
        
        risk_counts = deviation_risks['Deviation_Risk'].value_counts()
        
        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title='AI Risk Prediction Analysis',
            labels={'x': 'Risk Level', 'y': 'Number of Pallets'},
            color=risk_counts.index,
            color_discrete_map={
                'High Risk': 'red',
                'Medium Risk': 'orange',
                'Low Risk': 'green'
            }
        )
        return fig
    
    def _create_eta_analysis(self):
        """Create ETA prediction visualization"""
        eta_predictions = self.tracker.ml_system.predict_eta(self.tracker.ongoing_trips)
        
        route_etas = pd.DataFrame({
            'Route': self.tracker.ongoing_trips['Nearest_Route'],
            'Hours_to_Arrival': eta_predictions['Estimated_Hours_to_Arrival']
        })
        
        fig = px.box(
            route_etas,
            x='Route',
            y='Hours_to_Arrival',
            title='AI-Powered Delivery Time Predictions',
            labels={
                'Route': 'Route ID',
                'Hours_to_Arrival': 'Estimated Hours to Arrival'
            }
        )
        return fig
    
    def _create_anomaly_detection(self):
        """Create anomaly detection visualization"""
        anomalies = self.tracker.ml_system.detect_anomalies(self.tracker.ongoing_trips)
        
        # Combine with route information
        anomaly_data = pd.DataFrame({
            'Route': self.tracker.ongoing_trips['Nearest_Route'],
            'Is_Anomaly': anomalies['Is_Anomaly'],
            'Anomaly_Score': anomalies['Anomaly_Score']
        })
        
        route_anomalies = anomaly_data.groupby('Route').agg({
            'Is_Anomaly': 'sum',
            'Anomaly_Score': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            route_anomalies,
            x='Anomaly_Score',
            y='Is_Anomaly',
            title='AI Anomaly Detection Results',
            labels={
                'Anomaly_Score': 'Average Anomaly Score',
                'Is_Anomaly': 'Number of Anomalies Detected'
            },
            hover_data=['Route']
        )
        return fig
    
    def save_visualizations(self, output_dir='outputs'):
        """Save all visualizations to files"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save interactive map
        m = self.create_interactive_map()
        m.save(f'{output_dir}/pallet_map.html')
        
        # Save dashboard plots
        figs = self.create_dashboard()
        for i, fig in enumerate(figs):
            fig.write_html(f'{output_dir}/dashboard_plot_{i+1}.html')

    def create_cluster_analysis(self, group_tracking):
        """Create detailed cluster analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Group Size Distribution',
                'Spatial Distribution',
                'Temporal Evolution',
                'Group Stability'
            )
        )
        
        if group_tracking is None or group_tracking.empty:
            # Add empty plots with "No Data Available" annotations
            for row in [1, 2]:
                for col in [1, 2]:
                    fig.add_annotation(
                        text="No Group Data Available",
                        xref="x domain", yref="y domain",
                        x=0.5, y=0.5,
                        showarrow=False,
                        row=row, col=col
                    )
        else:
            # Group Size Distribution
            group_sizes = group_tracking['Group_Size']
            fig.add_trace(
                go.Histogram(x=group_sizes, name='Group Sizes'),
                row=1, col=1
            )
            
            # Spatial Distribution
            fig.add_trace(
                go.Scatter(
                    x=group_tracking['Center_Longitude'],
                    y=group_tracking['Center_Latitude'],
                    mode='markers',
                    marker=dict(
                        size=group_tracking['Group_Size']*3,
                        color=group_tracking['Spread_KM'],
                        colorscale='Viridis'
                    ),
                    name='Groups'
                ),
                row=1, col=2
            )
            
            # Temporal Evolution
            temporal_data = group_tracking.groupby('Timestamp').agg({
                'Group_Size': 'mean',
                'Spread_KM': 'mean'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=temporal_data['Timestamp'],
                    y=temporal_data['Group_Size'],
                    name='Avg Group Size'
                ),
                row=2, col=1
            )
            
            # Group Stability
            fig.add_trace(
                go.Scatter(
                    x=temporal_data['Timestamp'],
                    y=temporal_data['Spread_KM'],
                    name='Avg Spread'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Cluster Analysis Dashboard",
            showlegend=True
        )
        
        return fig
        
    def create_route_adherence_analysis(self, deviation_analysis):
        """Create detailed route adherence analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "xy"}, {"type": "domain"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            subplot_titles=(
                'Deviation Distribution',
                'Severity Levels',
                'Return Time Analysis',
                'Direction Analysis'
            )
        )
        
        # Deviation Distribution
        fig.add_trace(
            go.Histogram(
                x=deviation_analysis['Deviation_Distance'],
                name='Deviations'
            ),
            row=1, col=1
        )
        
        # Severity Levels
        severity_counts = deviation_analysis['Severity'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                name='Severity'
            ),
            row=1, col=2
        )
        
        # Return Time Analysis
        if 'Estimated_Return_Time' in deviation_analysis.columns:
            fig.add_trace(
                go.Box(
                    y=deviation_analysis['Estimated_Return_Time'],
                    name='Return Time'
                ),
                row=2, col=1
            )
        else:
            fig.add_annotation(
                text="No Return Time Data Available",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=2, col=1
            )
        
        # Direction Analysis
        if 'Direction_To_Route' in deviation_analysis.columns:
            fig.add_trace(
                go.Histogram(
                    x=deviation_analysis['Direction_To_Route'],
                    name='Direction'
                ),
                row=2, col=2
            )
        else:
            fig.add_annotation(
                text="No Direction Data Available",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Route Adherence Analysis",
            showlegend=True
        )
        
        return fig
        
    def create_temporal_analysis(self, tracking_data):
        """Create temporal distribution analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Hourly Activity',
                'Daily Patterns',
                'Update Frequency',
                'Time Gap Analysis'
            )
        )
        
        # Hourly Activity
        hourly = pd.to_datetime(tracking_data['timestamp']).dt.hour.value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hourly.index, y=hourly.values, name='Hourly'),
            row=1, col=1
        )
        
        # Daily Patterns
        daily = pd.to_datetime(tracking_data['timestamp']).dt.dayofweek.value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], y=daily.values, name='Daily'),
            row=1, col=2
        )
        
        # Update Frequency
        update_gaps = tracking_data.groupby('Pallet_ID')['timestamp'].diff().dt.total_seconds() / 3600
        fig.add_trace(
            go.Histogram(x=update_gaps, name='Update Gaps'),
            row=2, col=1
        )
        
        # Time Gap Analysis
        fig.add_trace(
            go.Box(y=update_gaps, name='Gap Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Temporal Analysis Dashboard",
            showlegend=True
        )
        
        return fig 