import streamlit as st
import pandas as pd


st.title("Map Only Test")


STATION_COORDS = {
    "Kuching": {"lat": 1.5533, "lon": 110.3592},
    "Miri": {"lat": 4.3995, "lon": 113.9914},
    "Sibu": {"lat": 2.2873, "lon": 111.8305}
}


station = st.selectbox(
    "Select Station",
    ["Kuching", "Miri", "Sibu"]
)


map_data = pd.DataFrame(
    [STATION_COORDS[station]]
)


st.write(map_data)


st.map(
    map_data,
    zoom=10
)