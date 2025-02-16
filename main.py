from src.data_processing.data_analyzer import PalletTracker
from src.visualization.map_visualizer import PalletVisualizer
from src.ml.smart_route_predictor import SmartRoutePredictor
from src.ml.pallet_monitoring import PalletMonitor
from src.visualization.business_dashboard import BusinessDashboard
import os

def main():
    print("🚛 PalletSense Smart Tracking System")
    print("===================================")
    
    # Create output directory
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    # Initialize components
    tracker = PalletTracker()
    predictor = SmartRoutePredictor()
    monitor = PalletMonitor()
    visualizer = PalletVisualizer(tracker)
    dashboard = BusinessDashboard()
    
    # Load and analyze data
    print("\n📊 Loading and Analyzing Data...")
    tracker.load_data()
    
    # Smart Route Prediction
    print("\n🧠 Training Smart Route Prediction...")
    predictor.train_location_predictor(tracker.ongoing_trips)
    next_locations = predictor.predict_next_location(tracker.ongoing_trips)
    
    # Analyze deviations with severity levels
    print("\n🚨 Analyzing Route Deviations...")
    deviation_analysis = predictor.analyze_deviation(tracker.ongoing_trips, tracker.known_routes)
    
    # Pallet Loss Prevention
    print("\n⚠️ Analyzing Stoppage Patterns...")
    stoppage_analysis = monitor.analyze_stoppages(tracker.ongoing_trips)
    loss_hotspots = monitor.identify_loss_hotspots()
    
    # Group Tracking
    print("\n🔍 Analyzing Pallet Groups...")
    group_tracking = monitor.track_pallet_groups(tracker.ongoing_trips)
    
    # Create visualizations
    print("\n📈 Generating Smart Tracking Dashboard...")
    
    # 1. Interactive Map
    print("- Creating interactive tracking map...")
    map_fig = visualizer.create_interactive_map()
    map_fig.save('outputs/smart_tracking_map.html')
    
    # 2. Route Performance Dashboard
    print("- Generating route performance analysis...")
    route_fig = dashboard.create_route_performance_dashboard(
        tracker.ongoing_trips,
        tracker.known_routes
    )
    route_fig.write_html('outputs/route_performance.html')
    
    # 3. Pallet Insights Dashboard
    print("- Creating pallet operations dashboard...")
    insights_fig = dashboard.create_pallet_insights_dashboard(
        tracker.ongoing_trips,
        stoppage_analysis,
        group_tracking
    )
    insights_fig.write_html('outputs/pallet_insights.html')
    
    # 4. Cluster Analysis
    print("- Generating cluster analysis visualization...")
    cluster_fig = visualizer.create_cluster_analysis(group_tracking)
    cluster_fig.write_html('outputs/dashboard_plot_1.html')
    
    # 5. Route Adherence Patterns
    print("- Analyzing route adherence patterns...")
    adherence_fig = visualizer.create_route_adherence_analysis(deviation_analysis)
    adherence_fig.write_html('outputs/dashboard_plot_2.html')
    
    # 6. Temporal Analysis
    print("- Creating temporal distribution analysis...")
    temporal_fig = visualizer.create_temporal_analysis(tracker.ongoing_trips)
    temporal_fig.write_html('outputs/dashboard_plot_3.html')
    
    # 7. Status Distribution
    print("- Generating status distribution chart...")
    status_fig = dashboard.create_status_distribution(tracker.ongoing_trips)
    status_fig.write_html('outputs/pallet_status.html')
    
    # 8. Statistical Distributions
    print("- Creating statistical distribution analysis...")
    dist_fig = dashboard.create_distribution_analysis(
        stoppage_analysis,
        group_tracking,
        tracker.ongoing_trips
    )
    dist_fig.write_html('outputs/pallet_distributions.html')
    
    # 9. KPI Summary Dashboard
    print("- Generating KPI summary dashboard...")
    kpi_fig = dashboard.create_kpi_dashboard(
        tracker.ongoing_trips,
        deviation_analysis,
        stoppage_analysis
    )
    kpi_fig.write_html('outputs/kpi_summary.html')
    
    # 10. Revenue Impact Analysis
    print("- Creating revenue impact analysis...")
    revenue_fig = dashboard.create_revenue_analysis(
        tracker.ongoing_trips,
        deviation_analysis
    )
    revenue_fig.write_html('outputs/revenue_impact.html')
    
    # Generate KPI Summary
    kpi_summary = dashboard.create_kpi_summary(
        tracker.ongoing_trips,
        deviation_analysis,
        stoppage_analysis
    )
    
    # Print summary
    print("\n✨ Analysis Complete!")
    print("\n📊 Key Performance Indicators:")
    for kpi, value in kpi_summary.items():
        print(f"- {kpi}: {value}")
    
    print("\n📍 Check the outputs directory for:")
    print("\nMain Visualizations:")
    print("1. Smart Tracking Map (smart_tracking_map.html)")
    print("   - Live pallet locations with risk indicators")
    print("   - Predicted next locations")
    print("   - Loss hotspots and group tracking")
    
    print("\n2. Route Performance Dashboard (route_performance.html)")
    print("   - Route adherence with target thresholds")
    print("   - Delay patterns with severity indicators")
    print("   - Revenue impact analysis")
    
    print("\n3. Pallet Operations Dashboard (pallet_insights.html)")
    print("   - Status distribution with color-coding")
    print("   - Stoppage patterns with warning thresholds")
    print("   - Group tracking analysis")
    
    print("\nDetailed Analytics:")
    print("4. Cluster Analysis (dashboard_plot_1.html)")
    print("5. Route Adherence Patterns (dashboard_plot_2.html)")
    print("6. Temporal Analysis (dashboard_plot_3.html)")
    print("7. Status Distribution (pallet_status.html)")
    print("8. Statistical Distributions (pallet_distributions.html)")
    
    print("\nKPI Dashboards:")
    print("9. KPI Summary (kpi_summary.html)")
    print("10. Revenue Impact Analysis (revenue_impact.html)")

if __name__ == "__main__":
    main() 