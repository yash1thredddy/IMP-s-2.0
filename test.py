import os
import cProfile
import pstats
import time
import pandas as pd
import numpy as np
from rdkit import Chem
from memory_profiler import profile as memory_profile
import line_profiler
from functools import wraps
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import io
from contextlib import redirect_stdout
import tracemalloc
import gc
import psutil
import sys
from pathlib import Path
import json
from datetime import datetime

# Import functions to test from your application
from process_compounds import (
    get_chembl_ids,
    fetch_and_calculate,
    validate_smiles,
    extract_properties,
    calculate_efficiency_metrics,
    get_molecule_data,
    batch_fetch_activities,
    process_compound,
    get_classification
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("performance_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("performance_testing")

# Define test data directory
TEST_DATA_DIR = "test_data"
if not os.path.exists(TEST_DATA_DIR):
    os.makedirs(TEST_DATA_DIR)

RESULTS_DIR = "benchmark_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Test compounds with varying complexity
TEST_COMPOUNDS = [
    # Simple structure
    {
        "name": "QUERCETIN", 
        "smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
    },
    # Medium complexity
    {
        "name": "KAEMPFEROL", 
        "smiles": "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"
    },
    # High complexity
    {
        "name": "BERBERINE", 
        "smiles": "COc1ccc2cc3[n+](cc2c1OC)CCc1cc2c(cc1-3)OCO2"
    }
]

# Generate a larger test dataset with a mix of valid and invalid SMILES
def generate_test_dataset(size=100):
    """Generate a dataset for large-scale testing"""
    np.random.seed(42)
    df = pd.DataFrame()
    
    # Get some valid SMILES from RDKit common molecules
    from rdkit import RDConfig
    from rdkit.Chem import rdmolfiles
    suppl = rdmolfiles.SDMolSupplier(os.path.join(RDConfig.RDDataDir, 'NCI', 'first_200.props.sdf'))
    valid_smiles = []
    valid_names = []
    
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        valid_smiles.append(Chem.MolToSmiles(mol))
        valid_names.append(f"compound_{i}")
        if len(valid_smiles) >= size // 2:
            break
    
    # Add some invalid SMILES
    invalid_smiles = ["C1=CC=C", "CC(", "C1=CC=CC=", "X", "1234", "C:1:C:C:1"] * (size // 12)
    invalid_names = [f"invalid_{i}" for i in range(len(invalid_smiles))]
    
    # Combine into dataframe
    df["compound_name"] = valid_names + invalid_names
    df["smiles"] = valid_smiles + invalid_smiles
    
    # Save to CSV
    csv_path = os.path.join(TEST_DATA_DIR, "test_compounds.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Generated test dataset with {len(df)} compounds saved to {csv_path}")
    return csv_path, df

# Create a wrapper for line_profiler to use directly as a decorator
def line_profile(follow=[]):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            prof = line_profiler.LineProfiler()
            prof.add_function(func)
            for f in follow:
                prof.add_function(f)
            try:
                with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull):
                        result = prof.runcall(func, *args, **kwargs)
                
                stream = io.StringIO()
                prof.print_stats(stream=stream)
                
                func_name = func.__name__
                filename = os.path.join(RESULTS_DIR, f"lineprofile_{func_name}.txt")
                with open(filename, 'w') as f:
                    f.write(stream.getvalue())
                
                logger.info(f"Line profile for {func_name} saved to {filename}")
                return result
            finally:
                del prof
        return wrapper
    return inner

# Function to run cProfile
def run_cprofile(func, *args, **kwargs):
    """Run cProfile on a function and save results"""
    func_name = func.__name__
    profile_output = os.path.join(RESULTS_DIR, f"cprofile_{func_name}.prof")
    stat_output = os.path.join(RESULTS_DIR, f"cprofile_{func_name}.txt")
    
    # Run the profiler
    cProfile.runctx(
        f'result = func(*args, **kwargs)', 
        globals(), 
        locals(),
        profile_output
    )
    
    # Print sorted stats
    stats = pstats.Stats(profile_output)
    stats.strip_dirs().sort_stats('cumulative').print_stats(20)
    
    # Save to file
    with open(stat_output, 'w') as f:
        stats = pstats.Stats(profile_output, stream=f)
        stats.strip_dirs().sort_stats('cumulative').print_stats(30)
    
    logger.info(f"cProfile results for {func_name} saved to {stat_output}")
    return locals()['result']

# Timer class for benchmarking
class Timer:
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        if self.name:
            logger.info(f"{self.name} took {self.elapsed:.6f} seconds")

# Memory tracker for memory usage
def track_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start tracing memory allocations
        tracemalloc.start()
        
        # Run garbage collection to clean up before measuring
        gc.collect()
        
        # Get process memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Get tracemalloc snapshot before
        snapshot1 = tracemalloc.take_snapshot()
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Get tracemalloc snapshot after
        snapshot2 = tracemalloc.take_snapshot()
        
        # Get process memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Log memory usage
        logger.info(f"Memory usage for {func.__name__}:")
        logger.info(f"  Process memory before: {memory_before:.2f} MB")
        logger.info(f"  Process memory after: {memory_after:.2f} MB")
        logger.info(f"  Process memory difference: {memory_after - memory_before:.2f} MB")
        
        # Log top memory blocks
        trace_file = os.path.join(RESULTS_DIR, f"memory_trace_{func.__name__}.txt")
        with open(trace_file, 'w') as f:
            f.write(f"Top memory allocations for {func.__name__}:\n")
            for stat in top_stats[:10]:
                f.write(f"{stat}\n")
        
        logger.info(f"Memory trace for {func.__name__} saved to {trace_file}")
        
        tracemalloc.stop()
        return result
    return wrapper

# Function to benchmark against different datasets
def benchmark_function(func, datasets, repeats=3):
    """Benchmark a function against multiple datasets"""
    results = []
    
    for dataset in datasets:
        dataset_name = dataset.get('name', 'unnamed')
        args = dataset.get('args', [])
        kwargs = dataset.get('kwargs', {})
        
        # Run multiple times to get average
        times = []
        for i in range(repeats):
            with Timer() as timer:
                func(*args, **kwargs)
            times.append(timer.elapsed)
        
        # Record results
        results.append({
            'function': func.__name__,
            'dataset': dataset_name,
            'min_time': min(times),
            'max_time': max(times),
            'avg_time': sum(times) / len(times)
        })
        
        logger.info(f"Benchmark {func.__name__} on {dataset_name}: avg={sum(times)/len(times):.6f}s")
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, f"benchmark_{func.__name__}.csv")
    df.to_csv(csv_path, index=False)
    
    return df

# Function to visualize benchmark results
def plot_benchmark_results(benchmark_df, title=None):
    """Create visualization for benchmark results"""
    if benchmark_df.empty:
        logger.warning("No benchmark data to plot")
        return
    
    func_name = benchmark_df['function'].iloc[0]
    
    plt.figure(figsize=(12, 6))
    
    # Bar plot for average times
    ax = sns.barplot(x='dataset', y='avg_time', data=benchmark_df)
    
    # Add error bars for min/max
    for i, row in benchmark_df.iterrows():
        ax.errorbar(i, row['avg_time'], 
                    yerr=[[row['avg_time']-row['min_time']], 
                          [row['max_time']-row['avg_time']]], 
                    fmt='o', color='black')
    
    plt.title(title or f"Performance Benchmark: {func_name}")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, f"benchmark_{func_name}_plot.png")
    plt.savefig(plot_path)
    logger.info(f"Benchmark plot saved to {plot_path}")
    plt.close()

# Create test cases for each function

def test_validate_smiles():
    """Test performance of SMILES validation"""
    # Generate test SMILES strings of varying complexity
    test_smiles = TEST_COMPOUNDS[:]
    
    # Add some invalid SMILES
    invalid_smiles = ["C1=CC=C", "CC(", "C1=CC=CC=", "X", "1234"]
    
    test_data = [
        {"name": "simple", "args": [TEST_COMPOUNDS[0]["smiles"]]},
        {"name": "complex", "args": [TEST_COMPOUNDS[2]["smiles"]]},
        {"name": "invalid", "args": [invalid_smiles[0]]}
    ]
    
    # Run line profiler
    @line_profile()
    def test_func(smiles):
        for _ in range(1000):  # Run multiple times to get meaningful profile
            validate_smiles(smiles)
    
    for case in test_data:
        test_func(case["args"][0])
    
    # Run cProfile
    run_cprofile(validate_smiles, TEST_COMPOUNDS[2]["smiles"])
    
    # Run benchmark
    benchmark_df = benchmark_function(validate_smiles, test_data, repeats=5)
    plot_benchmark_results(benchmark_df, "SMILES Validation Performance")
    
    return benchmark_df

def test_extract_properties():
    """Test performance of molecular property extraction"""
    test_data = [
        {"name": "simple", "args": [TEST_COMPOUNDS[0]["smiles"]]},
        {"name": "medium", "args": [TEST_COMPOUNDS[1]["smiles"]]},
        {"name": "complex", "args": [TEST_COMPOUNDS[2]["smiles"]]},
        {"name": "invalid", "args": ["C1=CC="]}
    ]
    
    # Run line profiler
    @line_profile()
    def test_func(smiles):
        for _ in range(100):  # Run multiple times to get meaningful profile
            extract_properties(smiles)
    
    for case in test_data:
        test_func(case["args"][0])
    
    # Run cProfile
    run_cprofile(extract_properties, TEST_COMPOUNDS[2]["smiles"])
    
    # Run memory profile
    @memory_profile
    def memory_test():
        for _ in range(100):
            for compound in TEST_COMPOUNDS:
                extract_properties(compound["smiles"])
    
    memory_test()
    
    # Run benchmark
    benchmark_df = benchmark_function(extract_properties, test_data, repeats=5)
    plot_benchmark_results(benchmark_df, "Property Extraction Performance")
    
    return benchmark_df

@track_memory
def test_get_chembl_ids():
    """Test performance of ChEMBL ID retrieval"""
    test_data = [
        {"name": "QUERCETIN", "args": [TEST_COMPOUNDS[0]["smiles"]], "kwargs": {"similarity_threshold": 20}},
        {"name": "KAEMPFEROL", "args": [TEST_COMPOUNDS[1]["smiles"]], "kwargs": {"similarity_threshold": 20}},
        {"name": "BERBERINE", "args": [TEST_COMPOUNDS[2]["smiles"]], "kwargs": {"similarity_threshold": 20}}
    ]
    
    # Mock Streamlit for testing
    import sys
    import importlib
    
    # Mock streamlit if needed
    if 'streamlit' not in sys.modules:
        class MockStreamlit:
            def __init__(self):
                pass
                
            def spinner(self, text):
                class SpinnerContextManager:
                    def __enter__(self):
                        return None
                    def __exit__(self, exc_type, exc_val, exc_tb):
                        pass
                return SpinnerContextManager()
                
            def error(self, text):
                pass
        
        sys.modules['streamlit'] = MockStreamlit()
    
    # Run cProfile
    for case in test_data:
        try:
            run_cprofile(
                get_chembl_ids, 
                case["args"][0], 
                case["kwargs"].get("similarity_threshold", 80)
            )
        except Exception as e:
            logger.error(f"Error testing get_chembl_ids with {case['name']}: {str(e)}")
    
    return None  # API dependent, can't reliably benchmark

@track_memory
def test_fetch_and_calculate():
    """Test performance of fetch_and_calculate function"""
    # This requires actual ChEMBL IDs, so we'll use a few common ones
    test_data = [
        {"name": "CHEMBL25", "args": ["CHEMBL25"]},  # Aspirin
        {"name": "CHEMBL521", "args": ["CHEMBL521"]},  # Ibuprofen
        {"name": "CHEMBL428647", "args": ["CHEMBL428647"]}  # Paclitaxel
    ]
    
    # Run cProfile for one case
    try:
        run_cprofile(fetch_and_calculate, "CHEMBL25")
    except Exception as e:
        logger.error(f"Error running cProfile on fetch_and_calculate: {str(e)}")
    
    return None  # API dependent, can't reliably benchmark

def test_batch_fetch_activities():
    """Test performance of batch activity fetching"""
    # This requires actual ChEMBL IDs
    chembl_ids = ["CHEMBL25", "CHEMBL521", "CHEMBL428647"]
    
    # Run cProfile
    try:
        run_cprofile(batch_fetch_activities, chembl_ids)
    except Exception as e:
        logger.error(f"Error testing batch_fetch_activities: {str(e)}")
    
    return None  # API dependent, can't reliably benchmark

def test_calculate_efficiency_metrics():
    """Test performance of efficiency metrics calculation"""
    # Generate test data variations
    test_cases = []
    
    # Normal values
    test_cases.append({
        "name": "normal_values",
        "args": [5.0, 100.0, 300.0, 4.0, 20]
    })
    
    # Edge cases
    test_cases.append({
        "name": "zero_values",
        "args": [5.0, 0.0, 300.0, 0.0, 20]
    })
    
    test_cases.append({
        "name": "nan_values",
        "args": [np.nan, 100.0, 300.0, 4.0, 20]
    })
    
    # Run line profiler
    @line_profile()
    def test_func(pActivity, psa, molecular_weight, npol, heavy_atoms):
        for _ in range(10000):  # Run many times to get meaningful profile
            calculate_efficiency_metrics(pActivity, psa, molecular_weight, npol, heavy_atoms)
    
    for case in test_cases:
        test_func(*case["args"])
    
    # Run cProfile
    run_cprofile(calculate_efficiency_metrics, *test_cases[0]["args"])
    
    # Run benchmark
    benchmark_df = benchmark_function(calculate_efficiency_metrics, test_cases, repeats=5)
    plot_benchmark_results(benchmark_df, "Efficiency Metrics Calculation Performance")
    
    return benchmark_df

def test_full_process():
    """Test full processing pipeline on a small set of compounds"""
    # Generate small test dataset
    csv_path, df = generate_test_dataset(size=10)
    
    # Test process_compound function on one compound
    sample_compound = df.iloc[0]
    
    try:
        # Run cProfile on process_compound
        run_cprofile(
            process_compound,
            compound_name=sample_compound['compound_name'],
            smiles=sample_compound['smiles'],
            similarity_threshold=80,
            activity_types=["IC50"]
        )
    except Exception as e:
        logger.error(f"Error testing process_compound: {str(e)}")
    
    return None  # Complex function with API calls, difficult to benchmark reliably

def create_timing_report():
    """Create final timing report with recommendations for C++ conversion"""
    timing_data = {}
    
    # Find all benchmark CSV files
    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('benchmark_') and f.endswith('.csv')]
    
    for csv_file in csv_files:
        func_name = csv_file.replace('benchmark_', '').replace('.csv', '')
        df = pd.read_csv(os.path.join(RESULTS_DIR, csv_file))
        
        if not df.empty:
            timing_data[func_name] = {
                'avg_time': df['avg_time'].mean(),
                'max_time': df['max_time'].max(),
                'datasets': len(df)
            }
    
    # Sort functions by average time (descending)
    sorted_funcs = sorted(timing_data.items(), key=lambda x: x[1]['avg_time'], reverse=True)
    
    # Create report
    report_path = os.path.join(RESULTS_DIR, "optimization_recommendations.txt")
    with open(report_path, 'w') as f:
        f.write("PERFORMANCE OPTIMIZATION RECOMMENDATIONS\n")
        f.write("=====================================\n\n")
        
        f.write("Functions ranked by execution time (slowest first):\n\n")
        
        for i, (func_name, stats) in enumerate(sorted_funcs):
            f.write(f"{i+1}. {func_name}\n")
            f.write(f"   Average time: {stats['avg_time']:.6f} seconds\n")
            f.write(f"   Maximum time: {stats['max_time']:.6f} seconds\n")
            f.write(f"   Tested on {stats['datasets']} datasets\n\n")
        
        f.write("\nRECOMMENDATIONS FOR C++ CONVERSION:\n")
        f.write("===============================\n\n")
        
        # Make recommendations based on timing data
        if sorted_funcs:
            # Recommend the top 3 slowest functions or fewer if less than 3
            top_n = min(3, len(sorted_funcs))
            f.write("Consider converting these functions to C++:\n\n")
            
            for i in range(top_n):
                func_name, stats = sorted_funcs[i]
                f.write(f"{i+1}. {func_name}\n")
                f.write(f"   Reason: This function takes {stats['avg_time']:.6f} seconds on average\n")
                
                # Add specific recommendations based on function name
                if 'extract_properties' in func_name:
                    f.write("   Note: This function involves heavy RDKit calculations which would benefit\n")
                    f.write("         significantly from C++ implementation.\n")
                elif 'validate_smiles' in func_name:
                    f.write("   Note: SMILES validation could be much faster in C++ and is called frequently.\n")
                elif 'calculate_efficiency_metrics' in func_name:
                    f.write("   Note: This math-heavy function would be faster in C++.\n")
                
                f.write("\n")
        else:
            f.write("No benchmark data available for making recommendations.\n")
        
        # Add profiling-based recommendations
        f.write("\nADDITIONAL INSIGHTS FROM PROFILING:\n")
        f.write("================================\n\n")
        
        # Check if cProfile results exist and add insights
        cprofile_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('cprofile_') and f.endswith('.txt')]
        if cprofile_files:
            f.write("Based on detailed profiling, these specific functions show high cumulative time:\n\n")
            
            for cprofile_file in cprofile_files[:3]:  # Consider top 3 cProfile results
                func_name = cprofile_file.replace('cprofile_', '').replace('.txt', '')
                f.write(f"- {func_name}: See detailed analysis in {cprofile_file}\n")
        
        # Add memory profiling insights
        memory_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('memory_trace_')]
        if memory_files:
            f.write("\nFunctions with high memory usage that might benefit from C++ optimization:\n\n")
            
            for memory_file in memory_files[:3]:
                func_name = memory_file.replace('memory_trace_', '').replace('.txt', '')
                f.write(f"- {func_name}: Memory usage details in {memory_file}\n")
    
    logger.info(f"Optimization recommendations report created at {report_path}")
    return report_path

# Main test execution
def run_all_tests():
    """Run all performance tests"""
    logger.info("Starting performance testing")
    
    # Make sure results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Run individual function tests
    logger.info("Testing SMILES validation...")
    test_validate_smiles()
    
    logger.info("Testing property extraction...")
    test_extract_properties()
    
    logger.info("Testing efficiency metrics calculation...")
    test_calculate_efficiency_metrics()
    
    # API-dependent tests (may be skipped in automated environments)
    if os.environ.get("RUN_API_TESTS", "False").lower() == "true":
        logger.info("Testing ChEMBL ID retrieval...")
        test_get_chembl_ids()
        
        logger.info("Testing fetch and calculate...")
        test_fetch_and_calculate()
        
        logger.info("Testing batch activity fetching...")
        test_batch_fetch_activities()
        
        logger.info("Testing full processing pipeline...")
        test_full_process()
    else:
        logger.info("Skipping API-dependent tests (set RUN_API_TESTS=True to enable)")
    
    # Create final report
    logger.info("Creating optimization recommendations report...")
    report_path = create_timing_report()
    
    logger.info(f"All tests completed. Results saved to {RESULTS_DIR}")
    logger.info(f"Optimization recommendations available at {report_path}")

if __name__ == "__main__":
    run_all_tests()