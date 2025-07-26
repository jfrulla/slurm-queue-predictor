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
    
    # Option 1: Run chunked extraction (memory-efficient, resumable)
    print("Running chunked extraction process...")
    extractor.run_extraction()
    
    # Option 2: Run with consolidation (creates single file + batch files)
    # extractor.run_extraction(consolidate=True)
    
    # Option 3: Step-by-step extraction (for more control)
    # extractor.connect_to_database()
    # batch_files = extractor.extract_job_data_chunked("custom_filename", "custom_checkpoint.json")
    # extractor.consolidate_batch_files(batch_files, "final_output.parquet")

if __name__ == "__main__":
    main() 