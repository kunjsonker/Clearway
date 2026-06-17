import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import pydeck as pdk
import os
import zipfile

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ClearWay AI | Predictive Engine",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 ClearWay AI: Predictive Dispatch Engine")
st.markdown("Transforming raw violation coordinates into prioritized operational targets using DBSCAN clustering.")

# -----------------------------------------------------------------------------
# STEP 1: DATA INGESTION & CACHING PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_and_clean_data(filepath):
    """
    Loads the dataset, calculates duration, and assigns vehicle impact weights.
    Cached so it only runs ONCE.
    """
    # Check if it's a zip file to handle Mac hidden files (__MACOSX) safely
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as z:
            # Find the actual CSV file, ignoring Mac hidden folders
            csv_files = [f for f in z.namelist() if f.endswith('.csv') and '__MACOSX' not in f]
            if not csv_files:
                raise ValueError("No valid CSV found in the zip file.")
            
            # Open only the clean CSV file
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
    else:
        df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False)
    
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce', utc=True)
    df['modified_datetime'] = pd.to_datetime(df['modified_datetime'], errors='coerce', utc=True)
    df = df.dropna(subset=['created_datetime', 'modified_datetime'])
    
    df['duration_mins'] = (df['modified_datetime'] - df['created_datetime']).dt.total_seconds() / 60.0
    df['duration_mins'] = df['duration_mins'].clip(lower=5, upper=240) 
    
    df['violation_clean'] = df['violation_type'].astype(str).str.replace(r'[\[\]\"]', '', regex=True)
    
    weight_map = {
        'TANKER': 10, 'LGV': 8, 'MAXI-CAB': 5, 'CAR': 4, 'VAN': 4,
        'GOODS AUTO': 3, 'PASSENGER AUTO': 2, 'SCOOTER': 1, 'MOTOR CYCLE': 1, 'MOPED': 1
    }
    df['vehicle_weight'] = df['vehicle_type'].map(weight_map).fillna(2)
    df['hour_of_day'] = df['created_datetime'].dt.hour
    df = df.dropna(subset=['latitude', 'longitude'])
    
    return df

# -----------------------------------------------------------------------------
# STEP 2: AI CLUSTERING & DASHBOARD UI
# -----------------------------------------------------------------------------
try:
    # --- DYNAMIC FILE LOADER ---
    # Look for the compressed version first to bypass GitHub's 100MB limit
    if os.path.exists("parking_data.csv.zip"):
        file_path = "parking_data.csv.zip"
    elif os.path.exists("parking_data.zip"):
        file_path = "parking_data.zip"
    else:
        file_path = "parking_data.csv"
        
    df = load_and_clean_data(file_path)
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("1. Temporal Filter")
    selected_hour = st.sidebar.slider("Current Hour of Day", 0, 23, 12)
    
    st.sidebar.header("2. 🕒 Next Hour — Historical Pattern")
    show_prediction = st.sidebar.checkbox("Show Historical Pattern for Next Hour", value=False)
    
    st.sidebar.header("3. Vehicle Focus")
    all_vehicles = df['vehicle_type'].dropna().unique().tolist()
    # By default, select everything.
    selected_vehicles = st.sidebar.multiselect("Filter by Vehicle Type", options=all_vehicles, default=all_vehicles)
    
    st.sidebar.header("4. AI Clustering Engine")
    st.sidebar.caption("Tune the algorithm's sensitivity.")
    eps_meters = st.sidebar.slider("Search Radius (meters)", 10, 200, 50, 10)
    min_samples = st.sidebar.slider("Min Incidents to form Hotspot", 2, 20, 3)

    # --- CORE FILTERING LOGIC ---
    # Determine which hour we are analyzing based on the Crystal Ball toggle
    target_hour = (selected_hour + 1) % 24 if show_prediction else selected_hour
    status_label = "Historical Pattern" if show_prediction else "Active"
    
    # Fast in-memory filter for the selected time AND selected vehicles
    filtered_df = df[(df['hour_of_day'] == target_hour) & (df['vehicle_type'].isin(selected_vehicles))].copy()
    
    # --- CORE CLUSTERING LOGIC ---
    if len(filtered_df) > 0:
        # Mathematically precise distance calculation using Haversine (Earth's curvature)
        kms_per_radian = 6371.0088
        eps_radians = (eps_meters / 1000.0) / kms_per_radian
        
        # Run DBSCAN on spatial coordinates converted to radians
        coords_radians = np.radians(filtered_df[['latitude', 'longitude']].values)
        db = DBSCAN(eps=eps_radians, min_samples=min_samples, metric='haversine', algorithm='ball_tree').fit(coords_radians)
        filtered_df['cluster_id'] = db.labels_
        
        # Separate the hotspots from the isolated noise
        hotspots_df = filtered_df[filtered_df['cluster_id'] != -1]
        noise_df = filtered_df[filtered_df['cluster_id'] == -1]
        
        # --- IMPACT SCORING ---
        if not hotspots_df.empty:
            # Aggregate data by cluster to score them
            cluster_metrics = hotspots_df.groupby('cluster_id').agg(
                center_lat=('latitude', 'mean'),
                center_lon=('longitude', 'mean'),
                total_violations=('cluster_id', 'count'),
                avg_duration=('duration_mins', 'mean'),
                heavy_count=('vehicle_weight', lambda x: sum(x >= 5)), # TANKERS/CABS
                total_weight=('vehicle_weight', 'sum'),
                common_violation=('violation_clean', lambda x: x.mode().iloc[0] if not x.empty else "MIXED"),
                common_location=('location', lambda x: x.mode().iloc[0] if not x.empty else "Unknown Area")
            ).reset_index()
            
            # The Magic Metric: Traffic Impact Score
            cluster_metrics['impact_score'] = round(
                cluster_metrics['total_weight'] * (1 + (cluster_metrics['avg_duration'] / 60)), 1
            )
            
            # Sort by highest priority
            cluster_metrics = cluster_metrics.sort_values(by='impact_score', ascending=False).reset_index(drop=True)
            
            # --- TOP METRICS ---
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{status_label} Hotspots Detected", len(cluster_metrics))
            col2.metric("Total Vehicles in Hotspots", len(hotspots_df))
            col3.metric("Isolated Incidents Ignored", len(noise_df))
            
            st.markdown("---")
            
            # --- CITY-WIDE TRENDS ---
            st.subheader("📊 Daily Congestion Trends")
            st.caption("City-wide volume of violations by hour based on your current vehicle filters.")
            # Filter the main dataframe by selected vehicles to show accurate trends
            trend_df = df[df['vehicle_type'].isin(selected_vehicles)]
            hourly_counts = trend_df.groupby('hour_of_day').size()
            st.bar_chart(hourly_counts, color="#ff4b4b")
            
            st.markdown("---")
            
            # --- 3D MAP RENDERING (PYDECK) ---
            st.subheader(f"🗺️ 3D Operational Heatmap ({target_hour}:00)")
            st.caption(f"Showing **{status_label}** target zones. Height and color represent Congestion Impact Scores. *(Hold Shift and drag to tilt map)*")
            
            # Prepare map data
            map_data = cluster_metrics[['cluster_id', 'center_lat', 'center_lon', 'impact_score', 'common_location']].rename(
                columns={'center_lat': 'lat', 'center_lon': 'lon'}
            )
            
            # Calculate dynamic center
            mid_lat, mid_lon = map_data['lat'].mean(), map_data['lon'].mean()
            
            # Build PyDeck 3D Layer
            view_state = pdk.ViewState(
                latitude=mid_lat if not pd.isna(mid_lat) else 12.97,
                longitude=mid_lon if not pd.isna(mid_lon) else 77.59,
                zoom=11.5,     # Increased zoom to bring the city closer
                pitch=55,      # Increased pitch for a much stronger 3D perspective
            )
            
            column_layer = pdk.Layer(
                'ColumnLayer',
                data=map_data,
                get_position='[lon, lat]',
                get_elevation='impact_score',
                elevation_scale=30,  # Doubled the height of the pillars
                radius=100,          # Reduced the thickness of the pillars
                get_fill_color='[255, 50, 50, 255]', # Made the color a solid, vibrant red
                pickable=True,
                auto_highlight=True,
            )
            
            deck = pdk.Deck(
                # Use CartoDB map style to bypass Mapbox API key requirement
                map_style='https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
                initial_view_state=view_state,
                layers=[column_layer],
                tooltip={"html": "<b>Hotspot Area:</b> {common_location} <br/> <b>Impact Score:</b> {impact_score}", "style": {"color": "white"}}
            )
            
            # Explicit height to ensure it takes up enough space
            st.pydeck_chart(deck, use_container_width=True)
            
            st.markdown("---")
            
            # --- DISPATCH FEED & EXPORT ---
            col_header1, col_header2 = st.columns([0.7, 0.3])
            with col_header1:
                st.subheader(f"⚡ {status_label} Dispatch Priority Feed")
                # Added transparency to how the impact score is calculated!
                st.caption("**Impact Score Formula:** $\Sigma(\\text{vehicle weight}) \\times (1 + \\text{avg stagnation time} / 60\\text{min})$. <br> Vehicle weight reflects estimated lane-blocking severity; stagnation time reflects duration of road obstruction.", unsafe_allow_html=True)
            with col_header2:
                # Provide a downloadable CSV of the dispatch plan
                csv = cluster_metrics[['cluster_id', 'common_location', 'impact_score', 'total_violations', 'heavy_count', 'common_violation']].to_csv(index=False)
                st.download_button(
                    label="📥 Download Dispatch Plan (CSV)",
                    data=csv,
                    file_name=f"dispatch_plan_hour_{target_hour}.csv",
                    mime="text/csv",
                )
            
            # Show the top 5 worst hotspots
            for index, row in cluster_metrics.head(5).iterrows():
                with st.container():
                    # Truncate long addresses for cleaner UI
                    display_address = str(row['common_location']).split(', Bengaluru')[0]
                    
                    if index == 0:
                        st.error(f"🚨 **PRIORITY DISPATCH 1: {display_address}**")
                    else:
                        st.warning(f"⚠️ **PRIORITY {index + 1}: {display_address}**")
                        
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Impact Score", f"{row['impact_score']}")
                    c2.metric("Total Vehicles (Heavy)", f"{int(row['total_violations'])} ({int(row['heavy_count'])})")
                    c3.metric("Avg Stagnation", f"{int(row['avg_duration'])} mins")
                    c4.write(f"**Primary Issue:**\n{row['common_violation']}")
                    st.write("")
        else:
            st.info("No dense clusters found with current settings. Try increasing search radius or decreasing min incidents.")

    else:
        st.warning(f"No traffic incidents recorded for {target_hour}:00 matching the current filters.")

except Exception as e:
    st.error(f"⚠️ Error loading data: {e}. Please ensure your dataset is located in the root repository folder and is named `parking_data.csv.zip`, `parking_data.zip`, or `parking_data.csv`.")