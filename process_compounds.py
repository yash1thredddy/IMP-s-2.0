import os
import shutil
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import QED
from chembl_webresource_client.new_client import new_client
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors, Crippen
import requests
from functools import lru_cache
import warnings
import streamlit as st
from typing import Dict, List, Optional, Union, Tuple
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add these lines to suppress chembl client logs
chembl_logger = logging.getLogger('chembl_webresource_client')
chembl_logger.setLevel(logging.WARNING)  # Only show warnings and errors

# Directory Configuration
RESULTS_DIR = "analysis_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# API and Processing Constants
ACTIVITY_TYPES = ["IC50", "EC50", "Ki", "Kd", "AC50", "GI50", "MIC"]
MAX_CSV_SIZE_MB = 10
MAX_BATCH_SIZE = 50
API_TIMEOUT = 30
MAX_RETRIES = 3




# Retry Configuration
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
RETRY_BACKOFF_FACTOR = 1
# Move this line after the retry strategy is defined
retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=RETRY_BACKOFF_FACTOR,
    status_forcelist=RETRY_STATUS_CODES,
)
adapter = HTTPAdapter(max_retries=retry_strategy)

# Initialize ChEMBL client with retry
similarity = new_client.similarity
molecule = new_client.molecule
activity = new_client.activity

# Setup session for API calls
session = requests.Session()
session.mount("http://", adapter)
session.mount("https://", adapter)

def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string using RDKit.
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(smiles, str):
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def validate_compound_name(name: str) -> bool:
    """
    Validate compound name.
    
    Args:
        name: Compound name to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(name, str):
        return False
    
    # Check length
    if len(name) < 1 or len(name) > 100:
        return False
    
    # Check for invalid characters
    invalid_chars = '<>:"/\\|?*'
    if any(char in name for char in invalid_chars):
        return False
    
    return True

# Ensure `analysis_results` directory exists

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
def get_molecule_data(chembl_id: str) -> Optional[Dict]:
    """
    Fetch molecule data directly from API.
    
    Args:
        chembl_id: ChEMBL ID to fetch
    
    Returns:
        Optional[Dict]: Molecule data or None if error
    """
    try:
        return molecule.get(chembl_id)
    except Exception as e:
        logger.error(f"Error fetching molecule data for {chembl_id}: {str(e)}")
        return None

def get_classification(inchikey: str) -> Optional[Dict]:
    """Get classification data from API"""
    try:
        url = f'http://classyfire.wishartlab.com/entities/{inchikey}.json'
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.error(f"Error getting classification for {inchikey}: {str(e)}")
        return None

# Block 2: Basic ChEMBL Functions
def get_chembl_ids(smiles: str, similarity_threshold: int = 80) -> List[Dict[str, str]]:
    """
    Perform similarity search with improved error handling.
    
    Args:
        smiles: SMILES string to search
        similarity_threshold: Similarity threshold (0-100)
    
    Returns:
        List[Dict[str, str]]: List of ChEMBL IDs
    """
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")
    
    try:
        with st.spinner("Performing similarity search..."):
            results = similarity.filter(
                smiles=smiles,
                similarity=similarity_threshold
            ).only(['molecule_chembl_id'])
            
            return [{"ChEMBL ID": result['molecule_chembl_id']} for result in results]
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        st.error(f"Error performing similarity search: {str(e)}")
        return []




def extract_classification_data(classification_result):
    """Extract classification fields with safe access"""
    if classification_result is None:
        return {
            'Kingdom': '',
            'Superclass': '',
            'Class': '',
            'Subclass': ''
        }
    
    try:
        return {
            'Kingdom': classification_result.get('kingdom', {}).get('name', '') if classification_result.get('kingdom') else '',
            'Superclass': classification_result.get('superclass', {}).get('name', '') if classification_result.get('superclass') else '',
            'Class': classification_result.get('class', {}).get('name', '') if classification_result.get('class') else '',
            'Subclass': classification_result.get('subclass', {}).get('name', '') if classification_result.get('subclass') else ''
        }
    except Exception as e:
        print(f"Error extracting classification data: {str(e)}")
        return {
            'Kingdom': '',
            'Superclass': '',
            'Class': '',
            'Subclass': ''
        }
# Block 3: Property Calculation Functions
def extract_properties(smiles):
    """Extract molecular properties from SMILES"""
    if smiles == 'N/A':
        return np.nan, np.nan, np.nan
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan, np.nan, np.nan
            
        hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
        hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        
        return hbd, hba, heavy_atoms
    except:
        return np.nan, np.nan, np.nan

def calculate_efficiency_metrics(pActivity, psa, molecular_weight, npol, heavy_atoms):
    """Calculate efficiency metrics with improved validation"""
    try:
        sei = pActivity / (psa / 100) if psa and not np.isnan(pActivity) and psa > 0 else np.nan
        bei = pActivity / (molecular_weight / 1000) if molecular_weight and not np.isnan(pActivity) and molecular_weight > 0 else np.nan
        nsei = pActivity / npol if npol and not np.isnan(pActivity) and npol > 0 else np.nan
        nbei = (npol * nsei + np.log10(heavy_atoms)) if npol and nsei and heavy_atoms and not np.isnan(nsei) and heavy_atoms > 0 else np.nan
        
        return sei, bei, nsei, nbei
    except Exception as e:
        logger.error(f"Error calculating efficiency metrics: {str(e)}")
        return np.nan, np.nan, np.nan, np.nan

# Block 4: Main Data Processing Function
def batch_fetch_activities(chembl_ids: List[str], batch_size: int = 5, max_retries: int = 3) -> List[Dict]:
    """
    Fetch activities in batches with retry logic.
    
    Args:
        chembl_ids: List of ChEMBL IDs
        batch_size: Size of each batch
        max_retries: Maximum number of retry attempts
    
    Returns:
        List[Dict]: List of activity data
    """
    if batch_size > MAX_BATCH_SIZE:
        logger.warning(f"Batch size {batch_size} exceeds maximum {MAX_BATCH_SIZE}. Using maximum value.")
        batch_size = MAX_BATCH_SIZE

    all_activities = []
    
    with st.spinner("Fetching activity data..."):
        progress_bar = st.progress(0)
        total_batches = len(range(0, len(chembl_ids), batch_size))
        
        for i in range(0, len(chembl_ids), batch_size):
            batch = chembl_ids[i:i + batch_size]
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    for activity_type in ACTIVITY_TYPES:
                        activities = activity.filter(
                            molecule_chembl_id__in=batch,
                            standard_type=activity_type
                        ).only('molecule_chembl_id', 'standard_value',
                              'standard_units', 'standard_type',
                              'target_chembl_id')
                        all_activities.extend(list(activities))
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Failed to fetch activities after {max_retries} attempts: {str(e)}")
                        st.warning(f"Some activity data could not be fetched for batch {i//batch_size + 1}")
                    time.sleep(1)  # Wait before retry
            
            # Update progress
            progress = (i + batch_size) / len(chembl_ids)
            progress_bar.progress(min(progress, 1.0))
    
    return all_activities

def fetch_and_calculate(chembl_id):
    """Fetch and calculate molecular properties with expanded activity types"""
    try:
        mol_data = get_molecule_data(chembl_id)
        if not mol_data:
            logger.warning(f"No molecule data found for {chembl_id}")
            return []
            
        molecular_properties = mol_data.get('molecule_properties', {})
        
        molecular_weight = float(molecular_properties.get('full_mwt', np.nan))
        psa = float(molecular_properties.get('psa', np.nan))
        
        smiles = mol_data.get('molecule_structures', {}).get('canonical_smiles', 'N/A')
        molecule_name = mol_data.get('pref_name', 'Unknown Name')

        # Extract molecular properties
        hbd, hba, heavy_atoms = extract_properties(smiles)
        npol = hbd + hba if not (np.isnan(hbd) or np.isnan(hba)) else np.nan

        # Generate InChIKey and get classification
        inchi_key = None
        if smiles != 'N/A':
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    inchi_key = Chem.MolToInchiKey(mol)
            except Exception as e:
                print(f"Error generating InChIKey for {chembl_id}: {str(e)}")

        classification_data = {}
        if inchi_key:
            classification_result = get_classification(inchi_key)
            classification_data = extract_classification_data(classification_result)
        else:
            classification_data = extract_classification_data(None)

        # Fetch all activity types
        results = []
        for activity_type in ACTIVITY_TYPES:
            bioactivities = list(activity.filter(
                molecule_chembl_id=chembl_id,
                standard_type=activity_type
            ).only('standard_value', 'standard_units', 'standard_type', 'target_chembl_id'))

            for act in bioactivities:
                if all(key in act for key in ['standard_value', 'standard_units', 'standard_type']):
                    if act['standard_value'] and act['standard_units'] == 'nM':
                        value = float(act['standard_value'])
                        pActivity = -np.log10(value * 1e-9)
                        
                        # Calculate efficiency metrics
                        sei, bei, nsei, nbei = calculate_efficiency_metrics(
                            pActivity, psa, molecular_weight, npol, heavy_atoms
                        )

                        # Calculate QED
                        qed = QED.qed(Chem.MolFromSmiles(smiles)) if smiles != 'N/A' else np.nan

                        results.append({
                            'ChEMBL ID': chembl_id,
                            'Molecule Name': molecule_name,
                            'SMILES': smiles,
                            'Molecular Weight': molecular_weight,
                            'TPSA': psa,
                            'Activity Type': activity_type,
                            'Activity (nM)': value,
                            'pActivity': pActivity,
                            'Target ChEMBL ID': act.get('target_chembl_id', ''),
                            'SEI': sei,
                            'BEI': bei,
                            'QED': qed,
                            'HBD': hbd,
                            'HBA': hba,
                            'Heavy Atoms': heavy_atoms,
                            'NPOL': npol,
                            'NSEI': nsei,
                            'nBEI': nbei,
                            'Kingdom': classification_data.get('Kingdom', ''),
                            'Superclass': classification_data.get('Superclass', ''),
                            'Class': classification_data.get('Class', ''),
                            'Subclass': classification_data.get('Subclass', '')
                        })

        if not results:  # If no activity data was found
            qed = QED.qed(Chem.MolFromSmiles(smiles)) if smiles != 'N/A' else np.nan
            results.append({
                'ChEMBL ID': chembl_id,
                'Molecule Name': molecule_name,
                'SMILES': smiles,
                'Molecular Weight': molecular_weight,
                'TPSA': psa,
                'Activity Type': 'Unknown',
                'Activity (nM)': np.nan,
                'pActivity': np.nan,
                'Target ChEMBL ID': '',
                'SEI': np.nan,
                'BEI': np.nan,
                'QED': qed,
                'HBD': hbd,
                'HBA': hba,
                'Heavy Atoms': heavy_atoms,
                'NPOL': npol,
                'NSEI': np.nan,
                'nBEI': np.nan,
                'Kingdom': classification_data.get('Kingdom', ''),
                'Superclass': classification_data.get('Superclass', ''),
                'Class': classification_data.get('Class', ''),
                'Subclass': classification_data.get('Subclass', '')
            })

        return results

    except Exception as e:
        print(f"Error processing ChEMBL ID {chembl_id}: {str(e)}")
        return []
    
# Block 5: Visualization Functions
def plot_all_visualizations(df_results, folder_name):
    """Generate all visualizations including original and new plots"""
    
    # Create subfolders
    sei_folder = os.path.join(folder_name, "SEI")
    bei_folder = os.path.join(folder_name, "BEI")
    activity_folder = os.path.join(folder_name, "Activity")
    
    for folder in [sei_folder, bei_folder, activity_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 1. SEI vs BEI scatter plot
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    unique_chembl_ids = df_results['ChEMBL ID'].unique()
    
    for chembl_id in unique_chembl_ids:
        df_subset = df_results[df_results['ChEMBL ID'] == chembl_id]
        ax1.scatter(df_subset['SEI'], df_subset['BEI'], alpha=0.6, label=chembl_id)
        ax1.plot(df_subset['SEI'], df_subset['BEI'], alpha=0.6)

    ax1.set_title('Scatter Plot of SEI vs BEI')
    ax1.set_xlabel('Surface Efficiency Index (SEI)')
    ax1.set_ylabel('Binding Efficiency Index (BEI)')
    ax1.grid(True)
    #ax1.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{folder_name}/sei_vs_bei_scatter_plot.png', dpi=300)
    plt.close(fig1)

    # 2. NSEI vs nBEI scatter plot
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    
    for chembl_id in unique_chembl_ids:
        df_subset = df_results[df_results['ChEMBL ID'] == chembl_id]
        ax2.scatter(df_subset['NSEI'], df_subset['nBEI'], alpha=0.6, label=chembl_id)
        ax2.plot(df_subset['NSEI'], df_subset['nBEI'], alpha=0.6)

    ax2.set_title('Scatter Plot of NSEI vs nBEI')
    ax2.set_xlabel('Normalized Surface Efficiency Index (NSEI)')
    ax2.set_ylabel('Normalized Binding Efficiency Index (nBEI)')
    ax2.grid(True)
    #ax2.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{folder_name}/nsei_vs_nbei_scatter_plot.png', dpi=300)
    plt.close(fig2)

    # 3. Activity Distribution box plot
    if 'Activity Type' in df_results.columns and 'pActivity' in df_results.columns:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_results, x='Activity Type', y='pActivity', ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        ax3.set_title('Activity Distribution by Type')
        plt.tight_layout()
        plt.savefig(f'{activity_folder}/activity_distribution.png', dpi=300)
        plt.close(fig3)

        # 4. Save separate activity plots for navigation in Streamlit
        activity_plots = [
            ("activity_vs_molecular_weight.png", "Activity vs Molecular Weight", "Molecular Weight"),
            ("activity_vs_QED.png", "Activity vs Drug-likeness (QED)", "QED"),
            ("activity_vs_NPOL.png", "Activity vs NPOL", "NPOL"),
            ("activity_class_distribution.png", "Distribution of Compound Superclasses", "Superclass")
        ]

        for filename, title, x_col in activity_plots:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if x_col == "Superclass":  # Bar plot for class distribution
                class_counts = df_results[x_col].value_counts()
                class_counts.plot(kind='bar', ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            else:  # Scatter plots for activities
                sns.scatterplot(data=df_results, x=x_col, y='pActivity', hue='Activity Type', ax=ax)

            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(f"{activity_folder}/{filename}", dpi=300)
            plt.close(fig)

    print("Activity visualizations saved successfully.")

def plot_property_with_structures(df, chembl_ids, property_name, title, group_index, folder_name):
    """Plot property distribution with molecular structures (clusters of 5)"""
    image_scale = 1
    fig_width = len(chembl_ids) * 3
    fig_height = 13
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plotting boxplot
    property_data = [df[df['ChEMBL ID'] == cid][property_name].tolist() for cid in chembl_ids]
    bp = ax.boxplot(property_data, patch_artist=True, positions=np.arange(len(chembl_ids)))

    # Customize boxplot colors
    for patch in bp['boxes']:
        patch.set_facecolor('#2196F3')
        patch.set_alpha(0.7)

    # Add individual points
    for i, cid in enumerate(chembl_ids):
        data = df[df['ChEMBL ID'] == cid][property_name]
        ax.scatter(np.repeat(i, len(data)), data, alpha=0.6, color='red', s=20)

    # Retrieving molecule data and names
    molecule_names = []
    molecules = []
    for cid in chembl_ids:
        mol_data = get_molecule_data(cid)
        if mol_data and mol_data['molecule_structures']:
            smiles = mol_data['molecule_structures']['canonical_smiles']
            mol = Chem.MolFromSmiles(smiles)
            molecules.append(mol)
            molecule_name = mol_data.get('pref_name') if mol_data.get('pref_name') else cid
        else:
            mol = None
            molecule_name = cid
        molecule_names.append(molecule_name)

    ax.set_xticks(np.arange(len(chembl_ids)))
    ax.set_xticklabels(molecule_names, rotation=45, ha='right')
    ax.set_title(f'{title} - Group {group_index + 1}')

    # Add molecular structures
    for i, (mol, name) in enumerate(zip(molecules, molecule_names)):
        if mol:
            img = Draw.MolToImage(mol, size=(int(200 * image_scale), int(200 * image_scale)))
            img_array = np.array(img)
            image_x = i / (len(chembl_ids) + 1)
            ax_image = fig.add_axes([image_x, 0.8, 0.1, 0.1], zorder=1)
            ax_image.imshow(img_array)
            ax_image.axis('off')
            ax_image.set_title(name, fontsize=8)

    plt.subplots_adjust(bottom=0.2, top=0.75, left=0.05, right=0.95)
    plt.savefig(f'{folder_name}/{property_name}_group{group_index + 1}_plot.png', 
                bbox_inches='tight', dpi=300)
    plt.close(fig)

# Block 6: Main Processing Functions
def process_compound(
    compound_name: str,
    smiles: str,
    similarity_threshold: int = 80
) -> Optional[pd.DataFrame]:
    """
    Process a single compound with improved error handling and progress tracking.
    
    Args:
        compound_name: Name of the compound
        smiles: SMILES string
        similarity_threshold: Similarity threshold for search
    
    Returns:
        Optional[pd.DataFrame]: Results dataframe or None if error
    """
    try:
        # Validate inputs
        if not validate_compound_name(compound_name):
            raise ValueError("Invalid compound name")
        if not validate_smiles(smiles):
            raise ValueError("Invalid SMILES string")
        
        compound_folder = os.path.join(RESULTS_DIR, compound_name.replace(' ', '_'))
        
        # Create directory structure
        for folder in [compound_folder,
                      os.path.join(compound_folder, "SEI"),
                      os.path.join(compound_folder, "BEI")]:
            os.makedirs(folder, exist_ok=True)
        
        # Fetch ChEMBL IDs
        with st.spinner("Searching for similar compounds..."):
            chembl_ids = get_chembl_ids(smiles, similarity_threshold)
            
            if not chembl_ids:
                st.warning("No similar compounds found")
                return None
            
            # Save ChEMBL IDs
            chembl_ids_df = pd.DataFrame(chembl_ids)
            chembl_ids_filename = os.path.join(compound_folder,
                                             f"{compound_name}_chembl_ids.csv")
            chembl_ids_df.to_csv(chembl_ids_filename, index=False)
        
        # Process compounds with progress tracking
        all_results = []
        with st.spinner("Processing compounds..."):
            progress_bar = st.progress(0)
            for idx, chembl_id_dict in enumerate(chembl_ids):
                chembl_id = chembl_id_dict['ChEMBL ID']
                results = fetch_and_calculate(chembl_id)
                all_results.extend(results)
                progress_bar.progress((idx + 1) / len(chembl_ids))
        
        # Create and save results DataFrame
        df_results = pd.DataFrame(all_results)
        df_results.replace("No data", np.nan, inplace=True)
        
        results_filename = os.path.join(compound_folder,
                                      f"{compound_name}_complete_results.csv")
        df_results.to_csv(results_filename, index=False)
        
        # Generate visualizations
        with st.spinner("Generating visualizations..."):
            plot_all_visualizations(df_results, compound_folder)
        
        st.success(f"Processing completed for {compound_name}")
        return df_results
    
    except Exception as e:
        logger.error(f"Error processing compound {compound_name}: {str(e)}")
        st.error(f"Error processing compound: {str(e)}")
        return None


def validate_csv_file(uploaded_file) -> bool:
    """
    Validate uploaded CSV file with improved column mapping.
    """
    try:
        # Make sure we can read from the file
        uploaded_file.seek(0)
        
        # Try reading the CSV
        df = pd.read_csv(uploaded_file)
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_CSV_SIZE_MB:
            st.error(f"File size exceeds maximum limit of {MAX_CSV_SIZE_MB}MB")
            return False

        # Show available columns
        st.write("Available columns:", list(df.columns))
        
        # Check required data is present
        if not any(col in df.columns for col in ['compound_name', 'compound']):
            st.error("CSV must contain either 'compound_name' or 'compound' column")
            return False
            
        if 'smiles' not in df.columns:
            st.error("CSV must contain 'smiles' column")
            return False

        # If we have 'compound' but not 'compound_name', use 'compound'
        if 'compound' in df.columns and 'compound_name' not in df.columns:
            df['compound_name'] = df['compound']

        # Validate data
        invalid_names = []
        invalid_smiles = []
        for idx, row in df.iterrows():
            compound_name = row.get('compound_name', row.get('compound', ''))
            if not validate_compound_name(str(compound_name).strip()):
                invalid_names.append(compound_name)
            if not validate_smiles(str(row['smiles']).strip()):
                invalid_smiles.append(idx + 1)
        
        if invalid_names:
            st.error(f"Invalid compound names found: {', '.join(invalid_names[:5])}")
            return False
        if invalid_smiles:
            st.error(f"Invalid SMILES strings found in rows: {invalid_smiles[:5]}")
            return False
        
        return True
    
    except Exception as e:
        st.error(f"Error validating CSV file: {str(e)}")
        logger.error(f"Error validating CSV file: {str(e)}")
        return False
    
    
# Function to check if a compound already exists and handle user choice
# Function to check if a compound already exists and handle user choice
def check_existing_compound(compound_name, smiles, similarity_threshold):
    """Check if the compound already exists and prompt the user for action."""
    compound_folder = os.path.join(RESULTS_DIR, compound_name.replace(' ', '_'))

    if os.path.exists(compound_folder):
        st.warning(f"⚠️ Compound **'{compound_name}'** already exists!")

        # Initialize session state variables if not set
        if "compound_action" not in st.session_state:
            st.session_state.compound_action = None
        if "new_compound_name" not in st.session_state:
            st.session_state.new_compound_name = ""
        if "confirm_choice" not in st.session_state:
            st.session_state.confirm_choice = False
        if "processing_triggered" not in st.session_state:
            st.session_state.processing_triggered = False

        # Create a form to prevent immediate updates
        with st.form("compound_confirmation_form", clear_on_submit=False):
            action = st.radio(
                "**What would you like to do?**",
                ["❌ Replace existing compound", "✏️ Enter a new compound name"],
                index=None,
                key="compound_action_radio"
            )

            # If "Enter a new compound name" is selected, show input box
            if action == "✏️ Enter a new compound name":
                new_name = st.text_input("Enter a new compound name:", key="new_name_input")
                st.session_state.new_compound_name = new_name  

            # Submit button for confirmation
            confirm = st.form_submit_button("✅ Confirm Selection")

            if confirm:
                if action:
                    st.session_state.compound_action = action
                    st.session_state.confirm_choice = True  
                    st.session_state.processing_triggered = False  
                    st.success("✔ Selection confirmed. Processing will proceed.")
                    st.experimental_rerun()  
                else:
                    st.error("Please select an option before confirming.")

        # Proceed with processing only after confirmation
        if st.session_state.confirm_choice and not st.session_state.processing_triggered:
            st.session_state.processing_triggered = True  

            if st.session_state.compound_action == "✏️ Enter a new compound name":
                new_compound_name = st.session_state.new_compound_name
                if new_compound_name:
                    st.success(f"✔ Processing with new name: **'{new_compound_name}'**")
                    process_compound(new_compound_name, smiles, similarity_threshold)
                    st.experimental_rerun()  
                    return new_compound_name  
                else:
                    st.error("Please enter a new compound name before confirming.")

            elif st.session_state.compound_action == "❌ Replace existing compound":
                shutil.rmtree(compound_folder)  
                st.success(f"✅ Replacing compound **'{compound_name}'** with new parameters.")
                process_compound(compound_name, smiles, similarity_threshold)
                st.experimental_rerun()  
                return compound_name  

        return None  

    return compound_name  


def process_and_store(
    compound_name: str,
    smiles: str,
    similarity_threshold: int = 80
) -> bool:
    """
    Process a compound and store results with improved validation.
    
    Args:
        compound_name: Name of the compound
        smiles: SMILES string
        similarity_threshold: Similarity threshold for search
    
    Returns:
        bool: True if processing successful, False otherwise
    """
    try:
        # Validate inputs
        if not validate_compound_name(compound_name):
            st.error("Invalid compound name. Please use alphanumeric characters and avoid special characters.")
            return False
        
        if not validate_smiles(smiles):
            st.error("Invalid SMILES string. Please check the input format.")
            return False
        
        # Check for existing compound
        validated_compound_name = check_existing_compound(compound_name, smiles, similarity_threshold)
        if validated_compound_name is None:
            return False
        
        # Process compound with progress tracking
        with st.spinner(f"Processing compound {validated_compound_name}..."):
            results = process_compound(validated_compound_name, smiles, similarity_threshold)
            
            if results is not None:
                st.success(f"Successfully processed {validated_compound_name}")
                st.session_state.processing_complete = True
                return True
            else:
                st.error(f"Failed to process {validated_compound_name}")
                st.session_state.processing_complete = False
                return False
    
    except Exception as e:
        logger.error(f"Error in process_and_store: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.session_state.processing_complete = False
        return False
