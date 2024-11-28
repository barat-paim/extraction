import streamlit as st
import pandas as pd
from data_store import BusDataStore
from datetime import datetime
import os

def init_data_store():
    try:
        store = BusDataStore()
        if store.store_bus_data():
            return store
        st.error("Failed to fetch bus data. Please try again.")
        return None
    except ValueError as e:
        st.error(str(e))  # This will show the API key missing error
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def main():
    st.title("Mercury - NYC Bus Tracker")
    
    # Initialize data store
    store = init_data_store()
    if not store:
        st.stop()  # Stop execution if store initialization failed

    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Natural Language", "Structured Query"]
    )

    if input_method == "Natural Language":
        query = st.text_input(
            "Ask about any bus (e.g., 'When is the next M57 bus at Madison Ave going east?')"
        )
        if query:
            # TODO: Add NLP processing
            st.info("Natural language processing coming soon!")
            
    else:
        col1, col2 = st.columns(2)
        with col1:
            bus_line = st.text_input("Bus Line (e.g., M57)")
        with col2:
            stop_name = st.text_input("Stop Name (e.g., MADISON)")
            
        if bus_line or stop_name:
            results = store.query_bus_location(bus_line, stop_name)
            if not results.empty:
                st.dataframe(results)
            else:
                st.warning("No buses found matching your criteria")

    # Auto-refresh
    if st.button("Refresh Data"):
        if store.store_bus_data():
            st.success("Data refreshed successfully!")
        else:
            st.error("Failed to refresh data")

if __name__ == "__main__":
    main() 