import os
import streamlit as st
import pandas as pd
import zipfile
import shutil
import glob
import matplotlib.pyplot as plt
from PIL import Image
from process_compounds import (
    process_and_store,
    validate_csv_file,
    validate_compound_name,
    validate_smiles,
    MAX_CSV_SIZE_MB
)
import logging
from typing import List, Optional, Dict
import time

# Add to app.py after imports
RESULTS_DIR = "analysis_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    state_vars = {
        "processing_complete": False,
        "compound_action": None,
        "new_compound_name": "",
        "confirm_choice": False,
        "error_state": None,
        "processing_progress": 0,
        "current_view": "main",
        "selected_plots": [],
        "batch_processing": False,
        "processing_compound": None,
        "compounds_to_process": [],
        "last_processed_compound": None,
        "show_new_compound_alert": False
    }
    
    for var, default in state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def reset_processing_state():
    """Reset all processing-related session state variables."""
    st.session_state.processing_complete = False
    st.session_state.compound_action = None
    st.session_state.confirm_choice = False
    st.session_state.error_state = None
    st.session_state.processing_progress = 0

def get_available_compounds() -> List[str]:
    """
    Get list of available processed compounds.
    
    Returns:
        List[str]: List of compound names
    """
    try:
        if not os.path.exists(RESULTS_DIR):
            return []
            
        compounds = sorted([d for d in os.listdir(RESULTS_DIR) 
                          if os.path.isdir(os.path.join(RESULTS_DIR, d))])
        logger.info(f"Found compounds: {compounds}")
        return compounds
    except Exception as e:
        logger.error(f"Error getting available compounds: {str(e)}")
        return []

def load_results(compound_name: str) -> Optional[pd.DataFrame]:
    """
    Load CSV results for the selected compound with error handling.
    
    Args:
        compound_name: Name of the compound
    
    Returns:
        Optional[pd.DataFrame]: Results dataframe or None if error
    """
    try:
        compound_name = compound_name.replace(" ", "_")
        file_path = os.path.join(RESULTS_DIR, compound_name, 
                                f"{compound_name}_complete_results.csv")
        
        if not os.path.exists(file_path):
            st.warning(f"⚠️ No results found for {compound_name}. CSV file is missing.")
            return None
        
        df = pd.read_csv(file_path)
        if df.empty:
            st.warning(f"⚠️ The results file for {compound_name} is empty.")
            return None
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading results for {compound_name}: {str(e)}")
        st.error(f"Error loading results: {str(e)}")
        return None

def zip_results() -> Optional[str]:
    """
    Create a zip file of all results with progress tracking.
    
    Returns:
        Optional[str]: Path to zip file or None if error
    """
    try:
        zip_filename = "processed_results.zip"
        compounds = get_available_compounds()
        
        if not compounds:
            st.warning("No results available to download.")
            return None
        
        with st.spinner("Creating ZIP file..."):
            progress_bar = st.progress(0)
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for idx, compound in enumerate(compounds):
                    compound_folder = os.path.join(RESULTS_DIR, compound)
                    for root, _, files in os.walk(compound_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, 
                                     os.path.relpath(file_path, RESULTS_DIR))
                    progress_bar.progress((idx + 1) / len(compounds))
            
            return zip_filename if os.path.exists(zip_filename) else None
    
    except Exception as e:
        logger.error(f"Error creating ZIP file: {str(e)}")
        st.error(f"Error creating download file: {str(e)}")
        return None
    
def zip_compound_results(compound_name: str) -> Optional[str]:
    """
    Create a zip file for a specific compound's results.
    
    Args:
        compound_name: Name of the compound to zip
    
    Returns:
        Optional[str]: Path to zip file or None if error
    """
    try:
        zip_filename = f"{compound_name}_results.zip"
        compound_folder = os.path.join(RESULTS_DIR, compound_name)
        
        if not os.path.exists(compound_folder):
            st.warning(f"No results available for {compound_name}.")
            return None
        
        with st.spinner(f"Creating ZIP file for {compound_name}..."):
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(compound_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, 
                                os.path.relpath(file_path, RESULTS_DIR))
            
            return zip_filename if os.path.exists(zip_filename) else None
    
    except Exception as e:
        logger.error(f"Error creating ZIP file for {compound_name}: {str(e)}")
        st.error(f"Error creating download file: {str(e)}")
        return None
    
def show_plots_with_navigation(folder_path: str, plot_type: str):
    """
    Display plots with navigation controls and error handling.
    
    Args:
        folder_path: Path to folder containing plots
        plot_type: Type of plot to display
    """
    try:
        if plot_type == "scatter":
            plot_files = ["sei_vs_bei_scatter_plot.png", 
                         "nsei_vs_nbei_scatter_plot.png"]
            state_key = "scatter_plot_index"
            button_key = "scatter"
        elif plot_type == "activity":
            plot_files = sorted(glob.glob(os.path.join(folder_path, "Activity", "*.png")))
            state_key = "activity_index"
            button_key = "activity"
        else:
            return
        
        if not plot_files:
            st.warning(f"No {plot_type} plots available.")
            return
        
        if state_key not in st.session_state:
            st.session_state[state_key] = 0
        
        # Navigation layout
        col1, col2, col3 = st.columns([1, 5, 1])
        with col1:
            if st.button("⬅ Previous", key=f"prev_{button_key}"):
                st.session_state[state_key] = (st.session_state[state_key] - 1) % len(plot_files)
        with col3:
            if st.button("Next ➡", key=f"next_{button_key}"):
                st.session_state[state_key] = (st.session_state[state_key] + 1) % len(plot_files)
        
        # Display plot
        current_index = st.session_state[state_key]
        if plot_type == "scatter":
            plot_path = os.path.join(folder_path, plot_files[current_index])
        else:
            plot_path = plot_files[current_index]
        
        if os.path.exists(plot_path):
            st.image(plot_path, 
                    caption=f"{plot_type.title()} Plot {current_index + 1}/{len(plot_files)}", 
                    use_container_width=True)
        else:
            st.warning(f"Plot file not found: {plot_path}")
    
    except Exception as e:
        logger.error(f"Error displaying {plot_type} plots: {str(e)}")
        st.error(f"Error displaying plots: {str(e)}")

def show_sei_bei_boxplots(compound_folder: str):
    """
    Display SEI/BEI box plots with improved navigation.
    
    Args:
        compound_folder: Path to compound folder
    """
    try:
        sei_folder = os.path.join(compound_folder, "SEI")
        bei_folder = os.path.join(compound_folder, "BEI")
        
        selected_property = st.radio("Select Property:", ["SEI", "BEI"])
        selected_folder = sei_folder if selected_property == "SEI" else bei_folder
        
        box_plots = sorted(glob.glob(os.path.join(selected_folder, f"{selected_property}_group*_plot.png")))
        
        if not box_plots:
            st.warning(f"No box plots available for {selected_property}.")
            return
        
        if "box_plot_index" not in st.session_state:
            st.session_state.box_plot_index = 0
        
        # Navigation
        col1, col2, col3 = st.columns([1, 5, 1])
        with col1:
            if st.button("⬅ Previous", key="prev_box"):
                st.session_state.box_plot_index = (st.session_state.box_plot_index - 1) % len(box_plots)
        with col3:
            if st.button("Next ➡", key="next_box"):
                st.session_state.box_plot_index = (st.session_state.box_plot_index + 1) % len(box_plots)
        
        # Display plot
        image_path = box_plots[st.session_state.box_plot_index]
        if os.path.exists(image_path):
            st.image(image_path, 
                    caption=f"{selected_property} Box Plot ({st.session_state.box_plot_index + 1}/{len(box_plots)})", 
                    use_container_width=True)
        else:
            st.warning(f"Plot file not found: {image_path}")
    
    except Exception as e:
        logger.error(f"Error displaying box plots: {str(e)}")
        st.error(f"Error displaying box plots: {str(e)}")

def main():
    """Main application function."""
    try:
        st.title("🔬 IMPULATOR")
    # Global progress indicator (always visible)
        if st.session_state.processing_compound:
            progress_container = st.container()
            with progress_container:
                st.info(f"⏳ Processing {st.session_state.processing_compound} in background...")
                st.progress(st.session_state.processing_progress)
        
        # Alert for newly processed compound
        if st.session_state.show_new_compound_alert:
            alert_container = st.container()
            with alert_container:
                new_compound = st.session_state.last_processed_compound
                st.success(f"✅ New compound processed: {new_compound}")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("View Results Now"):
                        compounds_list = get_available_compounds()
                        if new_compound in compounds_list:
                            st.session_state.selected_compound = new_compound
                            st.session_state.show_new_compound_alert = False
                            st.experimental_rerun()
                with col2:
                    if st.button("Dismiss"):
                        st.session_state.show_new_compound_alert = False
                        st.experimental_rerun()
        st.sidebar.header("Compound Processing")
        
        # Input method selection
        input_method = st.sidebar.radio("Input Method", ["Manual", "CSV Upload"])
        similarity_threshold = st.sidebar.slider("Similarity Threshold", 0, 100, 80)
        
        # Manual input processing
        if input_method == "Manual":
            compound_name = st.sidebar.text_input("Compound Name")
            smiles = st.sidebar.text_area("SMILES String")
            
            if st.sidebar.button("Process Compound"):
                if not compound_name or not smiles:
                    st.error("Please enter both compound name and SMILES.")
                    return
                
                if not validate_compound_name(compound_name):
                    st.error("Invalid compound name. Please use alphanumeric characters.")
                    return
                
                if not validate_smiles(smiles):
                    st.error("Invalid SMILES string. Please check the format.")
                    return
                
                with st.spinner("Processing compound... Please wait."):
                    process_and_store(compound_name, smiles, similarity_threshold)
        

        # CSV upload processing
# In the CSV upload processing section:
        elif input_method == "CSV Upload":
            uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
            
            if uploaded_file:
                try:
                    # Make sure we can read from the file
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.write(df.head())
                    
                    # Check required columns
                    required_columns = {'compound_name', 'smiles'}
                    if not all(col in df.columns for col in required_columns):
                        st.error(f"CSV must contain all required columns: {', '.join(required_columns)}")
                        return
                    
                    # Continue with processing if validation passes
                    st.sidebar.write("CSV Uploaded Successfully!")
                    
                    if st.sidebar.button("Process CSV"):
                        with st.spinner("Processing compounds... Please wait."):
                            progress_bar = st.progress(0)
                            for idx, row in df.iterrows():
                                process_and_store(
                                    compound_name=row['compound_name'],
                                    smiles=row['smiles'],
                                    similarity_threshold=similarity_threshold
                                )
                                progress_bar.progress((idx + 1) / len(df))
                            st.success("Processing completed for all compounds!")
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    logger.error(f"Error reading CSV file: {str(e)}")
        
        # Results display - FIXED VERSION
        st.sidebar.header("Select Processed Compound")
        
        # Get available compounds
        compounds_list = get_available_compounds()
        
        if compounds_list:
            selected_compound = st.sidebar.selectbox(
                "Choose a compound", 
                compounds_list
            )
            compound_folder = os.path.join(RESULTS_DIR, selected_compound)
            
            # Display results
            st.subheader(f"Results for: {selected_compound}")
            df_results = load_results(selected_compound)
            
            if df_results is not None and not df_results.empty:
                st.dataframe(df_results)
                
                # Create two columns for download options
                col1, col2 = st.columns(2)
                
                # CSV download option
                with col1:
                    csv_file = df_results.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV", 
                        csv_file, 
                        file_name=f"{selected_compound}_results.csv", 
                        mime="text/csv"
                    )
                
                # Zip download option for this compound
                with col2:
                    if st.button(f"📥 Download All {selected_compound} Files (ZIP)"):
                        with st.spinner(f"Preparing {selected_compound} files..."):
                            zip_file = zip_compound_results(selected_compound)
                            if zip_file:
                                with open(zip_file, "rb") as f:
                                    st.download_button(
                                        f"📥 Download {selected_compound} ZIP",
                                        f,
                                        file_name=zip_file,
                                        mime="application/zip"
                                    )

            # Display plots
            st.subheader("Scatter Plots")
            show_plots_with_navigation(compound_folder, "scatter")
            
            st.subheader("Activity Plots")
            show_plots_with_navigation(compound_folder, "activity")
            
            st.subheader("SEI & BEI Box Plots")
            show_sei_bei_boxplots(compound_folder)
            
            # Download all results
            st.sidebar.markdown("---")
            if st.sidebar.button("📥 Prepare All Results (ZIP)"):
                with st.sidebar.spinner("Creating ZIP of all results..."):
                    zip_file = zip_results()
                    if zip_file:
                        with open(zip_file, "rb") as f:
                            st.sidebar.download_button(
                                "📥 Download All Results (ZIP)",
                                f,
                                file_name=zip_file,
                                mime="application/zip"
                            )
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please try again or contact support.")
        reset_processing_state()

if __name__ == "__main__":
    init_session_state()
    main()