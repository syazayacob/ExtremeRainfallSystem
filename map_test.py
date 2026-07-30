import streamlit as st


st.set_page_config(
    page_title="Station Coordinate Test",
    layout="wide"
)


st.title("📍 Station Coordinate Test")


# Station coordinates
STATION_COORDS = {
    "Kuching": {
        "lat": 1.5533,
        "lon": 110.3592
    },
    "Miri": {
        "lat": 4.3995,
        "lon": 113.9914
    },
    "Sibu": {
        "lat": 2.2873,
        "lon": 111.8305
    }
}


# Station selection
station = st.selectbox(
    "Select Station",
    ["Kuching", "Miri", "Sibu"]
)


st.divider()

st.subheader(
    f"Station: {station}"
)


st.write(
    f"Latitude: {STATION_COORDS[station]['lat']}"
)

st.write(
    f"Longitude: {STATION_COORDS[station]['lon']}"
)