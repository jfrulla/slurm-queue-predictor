#!/usr/bin/env python3
"""
Example usage of the SLURM Data Extractor

This script demonstrates how to use the SlurmDataExtractor class
programmatically instead of via command line.
"""

from slurm_data_extractor import SlurmDataExtractor

def main():
    """Example extraction workflow."""
    
    # Initialize extractor with config file
    extractor = SlurmDataExtractor("config.yaml")
    
    # Option 1: Run complete extraction process
    print("Running complete extraction process...")
    extractor.run_extraction()
    
    # Option 2: Step-by-step extraction (for more control)
    # extractor.connect_to_database()
    # df = extractor.extract_job_data()
    # extractor.save_data(df, "custom_filename.parquet")

if __name__ == "__main__":
    main() 