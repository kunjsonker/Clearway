# 🚨 ClearWay AI: Predictive Traffic Dispatch Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clearway-qywccxncmsoyqpjaap57s3.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**ClearWay AI** is an operational intelligence and predictive dispatch dashboard built for traffic enforcement agencies. It shifts traffic policing from a **reactive, patrol-based model** to a **proactive, targeted model**.

By ingesting raw parking violation data, ClearWay AI mathematically identifies the most severe traffic bottlenecks, scores them based on estimated congestion impact, and generates actionable, real-time dispatch feeds for units on the ground.

---

## 🛑 The Problem: Blind Enforcement

Traffic enforcement today is largely reactive. Officers patrol randomly and often spend time on isolated, low-impact violations (e.g., a single scooter parked in an alleyway), while major arterial roads get choked by illegally parked heavy vehicles. **Sheer violation count does not equal congestion impact.**

## 💡 The Solution: Predictive Dispatch

Instead of viewing thousands of isolated data points, ClearWay AI groups violations into active **hotspots** and ranks them using a **Traffic Impact Score (TIS)**. The engine prioritizes clearing actual congestion over simply maximizing ticket counts, letting dispatchers send the right resources (e.g., tow trucks) to the right intersections first.

---

## 🚀 Core Engine & Methodology

### 1. Density-Based Spatial Clustering (DBSCAN)

Individual violation coordinates are grouped into active hotspots. To preserve accuracy over a real-world city grid, coordinates are converted to radians and clustered using the **Haversine metric**, which accounts for the Earth's spherical curvature rather than treating lat/long as flat Euclidean space.

### 2. Traffic Impact Score (TIS)

Not all violations carry equal weight. The engine combines the physical footprint of the vehicle involved with how long it has been obstructing the carriageway.

- **Vehicle Footprint:** Heavy vehicles (Tanker / LGV / Maxi-Cab) = weight 10–8, Cars/Vans = 4, Two-wheelers = 1.
- **Stagnation Duration:** How long the obstruction has persisted, derived from violation creation vs. resolution timestamps.

**Formula:**

$$\text{Impact Score} = \sum(\text{vehicle weight}) \times \left(1 + \frac{\text{avg stagnation time}}{60\text{min}}\right)$$

This is an explicit, domain-grounded proxy metric (not a model fitted against ground-truth congestion data, since no independent traffic-flow dataset was available) — it's designed so heavier, longer-blocking vehicles surface as higher priority than scattered minor violations.

### 3. Historical Pattern View ("Next Hour")

Toggling **Next Hour — Historical Pattern** doesn't predict the future from a trained model — it surfaces the aggregated historical pattern for the next hour, computed across the full Jan–May dataset. Since the underlying data spans months, this is a genuine multi-day aggregate rather than a single day's snapshot, giving dispatchers a grounded sense of what typically happens next so they can pre-position resources.

---

## ✨ Key Features

- **Interactive 3D Operational Heatmap:** Built with `pydeck`, visualizing impact scores as extruded 3D pillars on a dark-mode CartoDB map.
- **Live Dispatch Priority Feed:** Translates raw data into plain-English deployment targets with estimated severity.
- **City-Wide Congestion Analytics:** Bar charts tracking incident volume by hour, filterable by vehicle type.
- **One-Click Export:** Downloadable CSV dispatch plans for offline use by enforcement units.
- **Tunable Clustering:** Adjustable search radius and minimum-incidents threshold so the sensitivity of hotspot detection can be tuned live.

---

## 📂 Project Structure

```text
Clearway/
│
├── app.py                   # Main Streamlit application and clustering/scoring logic
├── parking_data.csv.zip     # Compressed violation dataset
├── requirements.txt         # Python dependencies (Streamlit, Pandas, scikit-learn, pydeck)
└── README.md                # Project documentation
```

---

## 🛠️ Local Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/kunjsonker/Clearway.git
cd Clearway
```

**2. Install the required dependencies**

```bash
pip install -r requirements.txt
```

**3. Ensure the data file is present**

Verify that `parking_data.csv.zip` is in the repository root. The app automatically extracts and reads from the compressed file (to stay under GitHub's 100MB file size limit). It will also fall back to `parking_data.zip` or `parking_data.csv` if present instead.

**4. Launch the dashboard**

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

The application is deployed on Streamlit Community Cloud:

👉 **[Launch ClearWay AI Dashboard](https://clearway-qywccxncmsoyqpjaap57s3.streamlit.app/)**

---

## ⚠️ Known Limitations

- The Impact Score is a designed heuristic, not a model validated against independent traffic-flow or congestion measurements — no such ground-truth dataset was available for this dataset.
- "Historical Pattern" forecasting reflects aggregated past behavior, not a trained time-series prediction.
- Vehicle weight values are estimated based on typical lane-blocking footprint and are adjustable in `app.py` (`weight_map`) as better calibration data becomes available.
