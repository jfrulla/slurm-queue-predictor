#!/usr/bin/env python3
"""
Example usage of the SLURM Data Preprocessor

This script demonstrates how to use the SlurmDataPreprocessor class
to preprocess raw SLURM batch files.
"""

from slurm_data_preprocessor import SlurmDataPreprocessor

def main():
    """Example preprocessing workflow."""
    
    # Initialize preprocessor with config file
    preprocessor = SlurmDataPreprocessor("config.yaml")
    
    # Option 1: Process all batch files matching a pattern
    print("Processing all batch files...")
    preprocessor.run_preprocessing(
        input_pattern="slurm_jobs_batch_*.parquet",
        output_filename="preprocessed_jobs.parquet"
    )
    
    # Option 2: Process only first few files for testing
    # preprocessor.run_preprocessing(
    #     input_pattern="slurm_jobs_batch_*.parquet",
    #     output_filename="test_preprocessed.parquet",
    #     max_files=5
    # )
    
    # Option 3: Process specific files
    # preprocessor.run_preprocessing(
    #     input_pattern="batch_000*.parquet",
    #     output_filename="subset_preprocessed.parquet"
    # )

if __name__ == "__main__":
    main() 