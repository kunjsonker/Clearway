🚨 ClearWay AI: Predictive Traffic Dispatch Engine
ClearWay AI is a predictive dispatch and operational intelligence dashboard built for traffic enforcement agencies. It shifts traffic policing from a reactive, patrol-based model to a proactive, targeted model.
By ingesting raw parking violation data, ClearWay AI mathematically identifies the most severe traffic bottlenecks, scores them based on actual congestion impact, and generates actionable, real-time dispatch feeds.
🛑 The Problem
Traffic enforcement today is largely blind and reactive. Officers patrol randomly and often waste time writing tickets for isolated, low-impact violations (like a scooter parked in an alley), while major arterial roads get completely choked by illegally parked heavy vehicles. Sheer violation count does not equal congestion impact.
💡 The Solution
Instead of viewing thousands of isolated data points, ClearWay AI groups violations into active hotspots and ranks them using a proprietary Traffic Impact Score (TIS). Our engine prioritizes clearing actual congestion over simply maximizing the number of tickets written.
🚀 Core Innovation & Methodology
Density-Based Spatial Clustering (DBSCAN)
We group individual violation coordinates into active "hotspots". To ensure mathematical precision over a city grid, the engine converts coordinates to radians and uses the Haversine metric to account for the Earth's spherical curvature.
Proprietary Traffic Impact Score
Not all violations are equal. Our engine weighs the physical footprint of the vehicles and how long they have been obstructing the road.
Vehicle Footprint: Heavy Vehicles (Tanker/LGV) = 10, Cars/Cabs = 4, Two-Wheelers = 1.
Stagnation Duration: The amount of time the bottleneck has persisted.
Formula: Σ(vehicle weight) × (1 + avg stagnation time / 60min)
Historical Pattern Forecasting
ClearWay AI doesn't just show current hotspots; it aggregates months of historical data to forecast multi-day historical patterns for the upcoming hour, allowing dispatchers to anticipate and prevent gridlock before it happens.
✨ Key Features
Interactive 3D Operational Heatmap: Built with Uber's pydeck, visualizing impact scores as extruded 3D pillars.
Live Dispatch Priority Feed: Translates data into plain-English deployment targets with estimated clearance needs (e.g., dispatching tow trucks vs. standard patrols).
City-Wide Congestion Analytics: Real-time bar charts tracking incident volumes by hour and vehicle type.
1-Click Export: Downloadable CSV Dispatch Plans for units on the ground.
🛠️ Tech Stack
Frontend/Dashboard: Streamlit
Data Processing: Pandas, NumPy
AI & Clustering Engine: Scikit-learn (DBSCAN)
3D Geospatial Visualization: PyDeck (deck.gl)
📦 Local Installation & Setup
Clone the repository
git clone [https://github.com/your-username/Clearway.git](https://github.com/your-username/Clearway.git)
cd Clearway


Install the required dependencies
pip install -r requirements.txt


Add the Dataset
Ensure your dataset is named parking_data.csv.zip (or parking_data.csv) and is placed in the root directory of this repository.
Run the Dashboard
streamlit run app.py


🌐 Live Demo
The application is currently deployed and live on Streamlit Community Cloud.
👉 Launch ClearWay AI
