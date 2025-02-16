import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

class BusinessDashboard:
    def __init__(self):
        self.kpi_thresholds = {
            'route_adherence': 0.9,    # 90% adherence target
            'update_frequency': 8,      # hours
            'utilization': 0.8         # 80% utilization target
        }
        
    def create_route_performance_dashboard(self, pallet_data, route_data):
        """Create comprehensive route performance analysis"""
        # Calculate route-specific metrics
        route_metrics = self._calculate_route_metrics(pallet_data)
        
        # Create subplots with better layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Route Adherence by Route',
                'Average Delays by Route',
                'Revenue Impact by Route',
                'Utilization by Route'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                  [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # 1. Route Adherence with target line
        adherence_bar = go.Bar(
            x=route_metrics.index,
            y=route_metrics['adherence_rate'] * 100,
            name='Route Adherence',
            marker_color='lightblue',
            hovertemplate="Route %{x}<br>Adherence: %{y:.1f}%<extra></extra>"
        )
        fig.add_trace(adherence_bar, row=1, col=1)
        fig.add_hline(y=90, line_dash="dash", line_color="red", 
                     annotation_text="Target (90%)", row=1, col=1)
        
        # 2. Average Delays with color gradient
        delays = route_metrics['avg_delay_hours']
        delay_colors = ['green' if x < 4 else 'orange' if x < 8 else 'red' for x in delays]
        delay_bar = go.Bar(
            x=route_metrics.index,
            y=delays,
            name='Average Delay',
            marker_color=delay_colors,
            hovertemplate="Route %{x}<br>Delay: %{y:.1f} hours<extra></extra>"
        )
        fig.add_trace(delay_bar, row=1, col=2)
        
        # 3. Revenue Impact with annotations
        revenue_bar = go.Bar(
            x=route_metrics.index,
            y=route_metrics['revenue_impact'],
            name='Revenue Impact',
            marker_color='red',
            hovertemplate="Route %{x}<br>Impact: $%{y:,.2f}<extra></extra>"
        )
        fig.add_trace(revenue_bar, row=2, col=1)
        
        # 4. Overall Utilization Gauge
        overall_util = route_metrics['utilization'].mean() * 100
        gauge = go.Indicator(
            mode="gauge+number",
            value=overall_util,
            title={'text': "Overall Fleet Utilization"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "red"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        )
        fig.add_trace(gauge, row=2, col=2)
        
        # Update layout with better styling
        fig.update_layout(
            height=800,
            title={
                'text': "Route Performance Dashboard",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': "Arial, sans-serif"}
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    
    def create_pallet_insights_dashboard(self, pallet_data, stoppage_analysis, group_tracking):
        """Create insights dashboard for pallet operations"""
        # Calculate pallet-specific metrics
        pallet_metrics = self._calculate_pallet_metrics(
            pallet_data, stoppage_analysis, group_tracking
        )
        
        # Create main figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Pallet Status Distribution',
                'Stoppage Duration Analysis',
                'Group Size Distribution',
                'Update Frequency Patterns'
            ),
            specs=[
                [{"type": "domain"}, {"type": "histogram"}],
                [{"type": "histogram"}, {"type": "histogram"}]
            ]
        )
        
        # 1. Enhanced Status Distribution
        status_colors = {
            'On Route': 'green',
            'Off Route': 'red',
            'Stationary': 'orange',
            'In Groups': 'blue'
        }
        fig.add_trace(
            go.Pie(
                labels=pallet_metrics['status_dist'].index,
                values=pallet_metrics['status_dist'].values,
                hole=0.4,
                marker_colors=[status_colors[x] for x in pallet_metrics['status_dist'].index],
                textinfo='percent+label',
                hovertemplate="Status: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # 2. Enhanced Stoppage Distribution with threshold regions
        warning_threshold = 12  # hours
        critical_threshold = 24  # hours
        
        fig.add_trace(
            go.Histogram(
                x=stoppage_analysis['Hours_Stationary'],
                nbinsx=20,
                name='Stoppage Duration',
                marker=dict(
                    color=['green' if x < warning_threshold else 
                          'orange' if x < critical_threshold else 'red' 
                          for x in stoppage_analysis['Hours_Stationary']],
                ),
                hovertemplate="Duration: %{x:.1f} hours<br>Count: %{y}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Add threshold annotations
        fig.add_annotation(
            x=warning_threshold,
            y=1,
            text="Warning (12h)",
            showarrow=True,
            arrowhead=2,
            row=1, col=2,
            yref='paper',
            ay=-40
        )
        fig.add_annotation(
            x=critical_threshold,
            y=1,
            text="Critical (24h)",
            showarrow=True,
            arrowhead=2,
            row=1, col=2,
            yref='paper',
            ay=-60
        )
        
        # 3. Enhanced Group Size Distribution
        if not group_tracking.empty:
            fig.add_trace(
                go.Histogram(
                    x=group_tracking['Group_Size'],
                    nbinsx=10,
                    name='Group Size',
                    marker_color='blue',
                    hovertemplate="Size: %{x} pallets<br>Count: %{y}<extra></extra>"
                ),
                row=2, col=1
            )
        
        # 4. Enhanced Update Frequency Distribution with threshold regions
        update_threshold = 8  # hours
        
        fig.add_trace(
            go.Histogram(
                x=pallet_metrics['update_frequencies'],
                nbinsx=20,
                name='Update Frequency',
                marker=dict(
                    color=['green' if x <= update_threshold else 'red' 
                          for x in pallet_metrics['update_frequencies']],
                ),
                hovertemplate="Hours: %{x:.1f}<br>Count: %{y}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Add update threshold annotation
        fig.add_annotation(
            x=update_threshold,
            y=1,
            text="Target Update (8h)",
            showarrow=True,
            arrowhead=2,
            row=2, col=2,
            yref='paper',
            ay=-40
        )
        
        # Update layout with better styling
        fig.update_layout(
            height=900,
            title={
                'text': "Pallet Operations Dashboard",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': "Arial, sans-serif"}
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # Add axis titles
        fig.update_xaxes(title_text="Hours", row=1, col=2)
        fig.update_xaxes(title_text="Group Size", row=2, col=1)
        fig.update_xaxes(title_text="Hours Since Update", row=2, col=2)
        fig.update_yaxes(title_text="Number of Pallets", row=1, col=2)
        fig.update_yaxes(title_text="Number of Groups", row=2, col=1)
        fig.update_yaxes(title_text="Number of Pallets", row=2, col=2)
        
        return fig
    
    def create_kpi_summary(self, pallet_data, route_metrics, stoppage_analysis):
        """Generate summary of key performance indicators"""
        kpis = {}
        
        # Overall route adherence
        kpis['overall_adherence'] = (
            sum(~pallet_data['Is_On_Route']) / len(pallet_data)
        )
        
        # Average update frequency
        kpis['avg_update_frequency'] = (
            pallet_data['Time_Since_Last_Update'].mean().total_seconds() / 3600
        )
        
        # Pallet utilization
        kpis['pallet_utilization'] = 1 - (
            len(stoppage_analysis[
                stoppage_analysis['Hours_Stationary'] > 24
            ]) / len(pallet_data)
        )
        
        # Revenue impact
        kpis['total_revenue_impact'] = pallet_data['Revenue_Impact'].sum()
        
        # Format KPIs for display
        summary = {
            'Route Adherence': f"{kpis['overall_adherence']*100:.1f}% "
                             f"({'✅' if kpis['overall_adherence'] >= self.kpi_thresholds['route_adherence'] else '❌'})",
            'Update Frequency': f"{kpis['avg_update_frequency']:.1f} hours "
                              f"({'✅' if kpis['avg_update_frequency'] <= self.kpi_thresholds['update_frequency'] else '❌'})",
            'Pallet Utilization': f"{kpis['pallet_utilization']*100:.1f}% "
                                 f"({'✅' if kpis['pallet_utilization'] >= self.kpi_thresholds['utilization'] else '❌'})",
            'Revenue Impact': f"${kpis['total_revenue_impact']:,.2f}"
        }
        
        return summary
    
    def _calculate_route_metrics(self, pallet_data):
        """Calculate detailed metrics for each route"""
        metrics = []
        
        for route_id, group in pallet_data.groupby('Nearest_Route'):
            # Calculate adherence rate
            adherence_rate = sum(group['Is_On_Route']) / len(group)
            
            # Calculate average delay
            avg_delay = group['Time_Since_Last_Update'].mean().total_seconds() / 3600
            
            # Calculate revenue impact
            revenue_impact = group['Revenue_Impact'].sum()
            
            # Calculate utilization (inverse of stationary time)
            utilization = 1 - (
                group['Time_Since_Last_Update'].dt.total_seconds().sum() /
                (len(group) * 24 * 3600)  # Total possible hours
            )
            
            metrics.append({
                'Route_ID': route_id,
                'adherence_rate': adherence_rate,
                'avg_delay_hours': avg_delay,
                'revenue_impact': revenue_impact,
                'utilization': utilization
            })
        
        return pd.DataFrame(metrics).set_index('Route_ID')
    
    def _calculate_pallet_metrics(self, pallet_data, stoppage_analysis, group_tracking):
        """Calculate detailed metrics for pallet operations"""
        metrics = {}
        
        # Status distribution
        metrics['status_dist'] = pd.Series({
            'On Route': sum(pallet_data['Is_On_Route']),
            'Off Route': sum(~pallet_data['Is_On_Route']),
            'Stationary': len(stoppage_analysis[
                stoppage_analysis['Hours_Stationary'] > 12
            ]),
            'In Groups': len(group_tracking['Pallets'].explode().unique())
                if not group_tracking.empty else 0
        })
        
        # Update frequencies
        metrics['update_frequencies'] = (
            pallet_data['Time_Since_Last_Update'].dt.total_seconds() / 3600
        )
        
        return metrics
    
    def create_status_distribution(self, pallet_data):
        """Create detailed status distribution visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "domain"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            subplot_titles=(
                'Current Status',
                'Status by Route',
                'Status Timeline',
                'Status Transitions'
            )
        )
        
        # Current Status
        status_counts = pd.Series({
            'On Route': sum(pallet_data['Is_On_Route']),
            'Off Route': sum(~pallet_data['Is_On_Route']),
            'Delayed': sum(pallet_data['Time_Since_Last_Update'].dt.total_seconds() / 3600 > 8),
            'Normal': sum(pallet_data['Time_Since_Last_Update'].dt.total_seconds() / 3600 <= 8)
        })
        
        fig.add_trace(
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                name='Current Status'
            ),
            row=1, col=1
        )
        
        # Status by Route
        route_status = pallet_data.groupby('Nearest_Route')['Is_On_Route'].value_counts().unstack()
        fig.add_trace(
            go.Bar(
                x=route_status.index,
                y=route_status[True],
                name='On Route'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(
                x=route_status.index,
                y=route_status[False],
                name='Off Route'
            ),
            row=1, col=2
        )
        
        # Status Timeline
        timeline_data = pallet_data.set_index('timestamp')['Is_On_Route'].resample('1H').mean()
        fig.add_trace(
            go.Scatter(
                x=timeline_data.index,
                y=timeline_data.values * 100,
                name='Route Adherence %'
            ),
            row=2, col=1
        )
        
        # Status Transitions
        update_freq = pallet_data.groupby('Pallet_ID')['Time_Since_Last_Update'].mean()
        fig.add_trace(
            go.Histogram(
                x=update_freq.dt.total_seconds() / 3600,
                name='Update Frequency'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Status Distribution Analysis",
            showlegend=True
        )
        
        return fig
        
    def create_distribution_analysis(self, stoppage_analysis, group_tracking, tracking_data):
        """Create comprehensive distribution analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Stoppage Duration',
                'Group Sizes',
                'Speed Distribution',
                'Update Frequency'
            )
        )
        
        # Stoppage Duration
        fig.add_trace(
            go.Histogram(
                x=stoppage_analysis['Hours_Stationary'],
                name='Stoppage Duration'
            ),
            row=1, col=1
        )
        
        # Group Sizes
        if not group_tracking.empty:
            fig.add_trace(
                go.Histogram(
                    x=group_tracking['Group_Size'],
                    name='Group Size'
                ),
                row=1, col=2
            )
        
        # Speed Distribution
        time_diffs = tracking_data.groupby('Pallet_ID')['timestamp'].diff().dt.total_seconds() / 3600
        distances = tracking_data.groupby('Pallet_ID').apply(
            lambda x: np.sqrt(
                (x['latitude'].diff()**2 + x['longitude'].diff()**2)
            ) * 111
        ).reset_index(level=0, drop=True)
        
        speeds = distances / time_diffs
        fig.add_trace(
            go.Histogram(
                x=speeds,
                name='Speed (km/h)'
            ),
            row=2, col=1
        )
        
        # Update Frequency
        update_freq = tracking_data.groupby('Pallet_ID')['Time_Since_Last_Update'].mean()
        fig.add_trace(
            go.Histogram(
                x=update_freq.dt.total_seconds() / 3600,
                name='Update Hours'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Distribution Analysis",
            showlegend=True
        )
        
        return fig
        
    def create_kpi_dashboard(self, tracking_data, deviation_analysis, stoppage_analysis):
        """Create KPI summary dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}]
            ],
            subplot_titles=(
                'Route Adherence',
                'Update Compliance',
                'Asset Utilization',
                'Risk Level'
            )
        )
        
        # Route Adherence
        adherence_rate = sum(tracking_data['Is_On_Route']) / len(tracking_data) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=adherence_rate,
                title={'text': "Route Adherence"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # Update Compliance
        update_compliance = sum(
            tracking_data['Time_Since_Last_Update'].dt.total_seconds() / 3600 <= 8
        ) / len(tracking_data) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=update_compliance,
                title={'text': "Update Compliance"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )
        
        # Asset Utilization
        utilization = (1 - len(stoppage_analysis) / len(tracking_data)) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=utilization,
                title={'text': "Asset Utilization"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 1, 'column': 0}
            ),
            row=2, col=1
        )
        
        # Risk Level
        high_risk = sum(deviation_analysis['Severity'].isin(['CRITICAL_DEVIATION', 'MEDIUM_DEVIATION']))
        risk_level = (1 - high_risk / len(deviation_analysis)) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_level,
                title={'text': "Risk Level"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 1, 'column': 1}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="KPI Dashboard",
            showlegend=False
        )
        
        return fig
        
    def create_revenue_analysis(self, tracking_data, deviation_analysis):
        """Create revenue impact analysis dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Revenue Impact by Route',
                'Impact vs Distance',
                'Cumulative Impact',
                'Risk Distribution'
            )
        )
        
        # Revenue Impact by Route
        route_impact = tracking_data.groupby('Nearest_Route')['Revenue_Impact'].sum()
        fig.add_trace(
            go.Bar(
                x=route_impact.index,
                y=route_impact.values,
                name='Revenue Impact'
            ),
            row=1, col=1
        )
        
        # Impact vs Distance
        fig.add_trace(
            go.Scatter(
                x=deviation_analysis['Deviation_Distance'],
                y=tracking_data['Revenue_Impact'],
                mode='markers',
                name='Impact vs Distance'
            ),
            row=1, col=2
        )
        
        # Cumulative Impact
        cumulative_impact = tracking_data['Revenue_Impact'].sort_values().cumsum()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative_impact))),
                y=cumulative_impact,
                name='Cumulative Impact'
            ),
            row=2, col=1
        )
        
        # Risk Distribution
        risk_impact = tracking_data.groupby(
            pd.cut(tracking_data['Revenue_Impact'], bins=5)
        ).size()
        fig.add_trace(
            go.Bar(
                x=[str(x) for x in risk_impact.index],
                y=risk_impact.values,
                name='Risk Distribution'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Revenue Impact Analysis",
            showlegend=True
        )
        
        return fig 