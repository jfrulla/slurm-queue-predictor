#!/usr/bin/env python3
"""
SLURM Data Preprocessor

This script preprocesses raw SLURM job data files extracted by slurm_data_extractor.py
to create features suitable for machine learning models.
"""

import os
import sys
import re
import logging
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


class SlurmDataPreprocessor:
    """Preprocess raw SLURM job data for machine learning."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            # Use default config if file not found
                         return {
                 'extraction': {
                     'output_dir': 'data/raw',
                     'output_format': 'parquet'
                 },
                 'preprocessing': {
                     'input_dir': 'data/raw',
                     'output_dir': 'data/preprocessed',
                     'n_jobs': -1,
                     'chunk_size': 1000000
                 },
                'logging': {
                    'level': 'INFO',
                    'log_to_file': True,
                    'log_file': 'logs/slurm_preprocessor.log'
                }
            }
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create logs directory if needed
        if log_config.get('log_to_file', False):
            log_file = log_config.get('log_file', 'logs/slurm_preprocessor.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler(sys.stdout)
                ]
            )
        else:
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
        
        self.logger = logging.getLogger(__name__)
    
    def parse_tres_field(self, tres_str: str) -> Dict[str, int]:
        """Parse TRES (Trackable RESource) field to extract resource requirements."""
        if pd.isna(tres_str) or not tres_str:
            return {'nodes': 0, 'gpus': 0, 'mem_mb': 0}
        
        resources = {'nodes': 0, 'gpus': 0, 'mem_mb': 0}
        
        # TRES format examples:
        # "cpu=8,mem=32G,node=1,billing=8"
        # "cpu=16,mem=64000M,node=2,gres/gpu=4"
        # "billing=1,cpu=1,mem=4000M,node=1"
        
        try:
            # Split by comma and process each resource
            for item in tres_str.split(','):
                if '=' in item:
                    key, value = item.split('=', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'node':
                        resources['nodes'] = int(value)
                    elif key in ['gres/gpu', 'gpu']:
                        # Handle GPU specifications
                        gpu_match = re.search(r'(\d+)', value)
                        if gpu_match:
                            resources['gpus'] = int(gpu_match.group(1))
                    elif key == 'mem':
                        # Handle memory specifications (convert to MB)
                        mem_value = self._parse_memory(value)
                        resources['mem_mb'] = mem_value
                        
        except (ValueError, AttributeError) as e:
            self.logger.debug(f"Error parsing TRES field '{tres_str}': {e}")
        
        return resources
    
    def _parse_memory(self, mem_str: str) -> int:
        """Parse memory string and convert to MB."""
        if not mem_str:
            return 0
        
        # Remove any whitespace
        mem_str = mem_str.strip().upper()
        
        # Extract number and unit
        match = re.match(r'(\d+(?:\.\d+)?)\s*([KMGT]?B?)', mem_str)
        if not match:
            return 0
        
        value = float(match.group(1))
        unit = match.group(2)
        
        # Convert to MB
        if unit in ['B', '']:
            return int(value / (1024 * 1024))
        elif unit in ['K', 'KB']:
            return int(value / 1024)
        elif unit in ['M', 'MB']:
            return int(value)
        elif unit in ['G', 'GB']:
            return int(value * 1024)
        elif unit in ['T', 'TB']:
            return int(value * 1024 * 1024)
        else:
            return int(value)  # Assume MB if unknown
    
    def calculate_wait_time(self, time_submit: int, time_start: int) -> int:
        """Calculate wait time in seconds."""
        if pd.isna(time_submit) or pd.isna(time_start) or time_start <= time_submit:
            return 0
        return int(time_start - time_submit)
    
    def process_single_batch(self, batch_file: Path) -> pd.DataFrame:
        """Process a single batch file and extract features."""
        self.logger.info(f"Processing batch file: {batch_file}")
        
        try:
            # Load the batch file
            if batch_file.suffix == '.parquet':
                df = pd.read_parquet(batch_file)
            elif batch_file.suffix == '.csv':
                df = pd.read_csv(batch_file)
            elif batch_file.suffix == '.json':
                df = pd.read_json(batch_file, orient='records')
            else:
                raise ValueError(f"Unsupported file format: {batch_file.suffix}")
            
            if df.empty:
                self.logger.warning(f"Empty batch file: {batch_file}")
                return pd.DataFrame()
            
            # Extract basic features
            processed_df = pd.DataFrame()
            
            # Job ID
            processed_df['job_id'] = df['id_job']
            
            # Basic job info
            processed_df['partition'] = df['partition']
            processed_df['account'] = df['account']
            processed_df['timelimit'] = df['timelimit']
            processed_df['cpus_req'] = df['cpus_req']
            processed_df['time_submit'] = df['time_submit']
            processed_df['time_start'] = df['time_start']
            processed_df['time_end'] = df['time_end']
            processed_df['state'] = df['state']
            
            # Calculate wait time
            processed_df['wait_time'] = df.apply(
                lambda row: self.calculate_wait_time(row['time_submit'], row['time_start']), 
                axis=1
            )
            
            # Parse TRES resources
            self.logger.info("Parsing TRES fields...")
            tres_data = df['tres_req'].apply(self.parse_tres_field)
            
            processed_df['nodes_req'] = [tres['nodes'] for tres in tres_data]
            processed_df['gpus_req'] = [tres['gpus'] for tres in tres_data]
            processed_df['mem_req_mb'] = [tres['mem_mb'] for tres in tres_data]
            
            # Sort by submit time for partition state calculations
            processed_df = processed_df.sort_values('time_submit').reset_index(drop=True)
            
            self.logger.info(f"Processed {len(processed_df)} jobs from {batch_file}")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing batch file {batch_file}: {e}")
            return pd.DataFrame()
    
    def get_cluster_state_with_sacct(self, start_time: int, end_time: int) -> pd.DataFrame:
        """Get cluster state using sacct command for the specified time range."""
        
        # Convert Unix timestamps to SLURM date format (YYYY-MM-DD)
        start_date = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime('%Y-%m-%d')
        end_date = datetime.fromtimestamp(end_time, tz=timezone.utc).strftime('%Y-%m-%d')
        
        self.logger.info(f"Querying cluster state from {start_date} to {end_date} using sacct...")
        
        # Build sacct command to get all jobs that were running during this period
        cmd = [
            'sacct',
            '--state=RUNNING,COMPLETED,CANCELLED,FAILED,TIMEOUT,NODE_FAIL',
            f'--starttime={start_date}',
            f'--endtime={end_date}',
            '--format=JobID,Partition,Account,Submit,Start,End,AllocCPUS,State,ReqNodes,ReqMem',
            '--parsable2',
            '--noheader',
            '--allocations'  # Only show main job allocations, not job steps
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.warning(f"sacct command failed: {result.stderr}")
                return pd.DataFrame()
            
            if not result.stdout.strip():
                self.logger.warning("No data returned from sacct")
                return pd.DataFrame()
            
            # Parse sacct output
            lines = result.stdout.strip().split('\n')
            data = []
            
            for line in lines:
                if not line.strip():
                    continue
                    
                fields = line.split('|')
                if len(fields) >= 10:
                    data.append({
                        'job_id': fields[0],
                        'partition': fields[1],
                        'account': fields[2], 
                        'submit_time': self._parse_slurm_time(fields[3]),
                        'start_time': self._parse_slurm_time(fields[4]),
                        'end_time': self._parse_slurm_time(fields[5]),
                        'alloc_cpus': self._parse_int_safe(fields[6]),
                        'state': fields[7],
                        'req_nodes': self._parse_int_safe(fields[8]),
                        'req_mem': fields[9]
                    })
            
            if not data:
                self.logger.warning("No valid job data parsed from sacct output")
                return pd.DataFrame()
            
            cluster_df = pd.DataFrame(data)
            self.logger.info(f"Retrieved {len(cluster_df)} jobs from cluster state query")
            return cluster_df
            
        except subprocess.TimeoutExpired:
            self.logger.error("sacct command timed out")
            return pd.DataFrame()
        except FileNotFoundError:
            self.logger.warning("sacct command not found - skipping cluster state calculation")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error running sacct: {e}")
            return pd.DataFrame()
    
    def _parse_slurm_time(self, time_str: str) -> int:
        """Parse SLURM time format to Unix timestamp."""
        if not time_str or time_str == 'Unknown':
            return 0
        
        try:
            # Handle different SLURM time formats
            if 'T' in time_str:
                # Format: 2024-01-15T10:30:00
                dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
            else:
                # Format: 2024-01-15 10:30:00
                dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            
            # Convert to UTC timestamp
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            return 0
    
    def _parse_int_safe(self, value_str: str) -> int:
        """Safely parse integer value from string."""
        if not value_str or value_str == 'Unknown':
            return 0
        try:
            return int(value_str)
        except ValueError:
            return 0
    
    def calculate_partition_state_at_submit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate partition state metrics using sacct for accurate cluster state."""
        self.logger.info("Calculating partition state at submit time using cluster data...")
        
        if df.empty:
            return df
        
        # Get time range for this batch
        min_submit = df['time_submit'].min()
        max_submit = df['time_submit'].max()
        
        # Add buffer to capture jobs that might have been running before first submit
        # and jobs that might still be running after last submit
        preprocessing_config = self.config.get('preprocessing', {})
        buffer_days = preprocessing_config.get('sacct_buffer_days', 7)
        buffer_seconds = buffer_days * 24 * 3600
        start_time = min_submit - buffer_seconds
        end_time = max_submit + buffer_seconds
        
        # Check if sacct should be used
        preprocessing_config = self.config.get('preprocessing', {})
        use_sacct = preprocessing_config.get('use_sacct', True)
        
        if not use_sacct:
            self.logger.info("sacct disabled in config - using fallback calculation")
            return self._calculate_partition_state_fallback(df)
        
        # Get cluster state using sacct
        cluster_state = self.get_cluster_state_with_sacct(start_time, end_time)
        
        if cluster_state.empty:
            self.logger.warning("No cluster state data available - using fallback calculation")
            return self._calculate_partition_state_fallback(df)
        
        # Initialize new columns
        df['jobs_running_at_submit'] = 0
        df['jobs_pending_at_submit'] = 0
        df['cores_allocated_at_submit'] = 0
        
        # Process each job in the batch
        for idx, job in tqdm(df.iterrows(), total=len(df), desc="Calculating partition state"):
            submit_time = job['time_submit']
            partition = job['partition']
            
            # Filter cluster state to same partition
            partition_jobs = cluster_state[cluster_state['partition'] == partition]
            
            if partition_jobs.empty:
                continue
            
            # Jobs running at submit time:
            # - Started before or at this job's submit time
            # - End time is after submit time or not set (still running)
            running_mask = (
                (partition_jobs['start_time'] <= submit_time) &
                (partition_jobs['start_time'] > 0) &  # Job actually started
                ((partition_jobs['end_time'] > submit_time) | (partition_jobs['end_time'] == 0))
            )
            
            # Jobs pending at submit time:
            # - Submitted before or at this job's submit time
            # - Not yet started at this job's submit time (start_time > submit_time or not set)
            pending_mask = (
                (partition_jobs['submit_time'] <= submit_time) &
                ((partition_jobs['start_time'] > submit_time) | (partition_jobs['start_time'] == 0))
            )
            
            # Calculate metrics
            running_jobs = partition_jobs[running_mask]
            pending_jobs = partition_jobs[pending_mask]
            
            df.loc[idx, 'jobs_running_at_submit'] = len(running_jobs)
            df.loc[idx, 'jobs_pending_at_submit'] = len(pending_jobs)
            df.loc[idx, 'cores_allocated_at_submit'] = running_jobs['alloc_cpus'].sum()
        
        self.logger.info("Partition state calculation completed using cluster data")
        return df
    
    def _calculate_partition_state_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback method when sacct is not available - uses only batch data."""
        self.logger.info("Using fallback partition state calculation (less accurate)")
        
        # Initialize new columns
        df['jobs_running_at_submit'] = 0
        df['jobs_pending_at_submit'] = 0
        df['cores_allocated_at_submit'] = 0
        
        # Group by partition for efficiency
        partitions = df['partition'].unique()
        
        for partition in tqdm(partitions, desc="Processing partitions (fallback)"):
            partition_mask = df['partition'] == partition
            partition_df = df[partition_mask].copy()
            
            if len(partition_df) == 0:
                continue
                
            # For each job in this partition, calculate state at submit time
            for idx, job in partition_df.iterrows():
                submit_time = job['time_submit']
                
                # Find all jobs in same partition
                same_partition = df['partition'] == partition
                
                # Jobs running at submit time (within this batch only)
                running_mask = (
                    same_partition &
                    (df['time_start'] <= submit_time) &
                    (df['time_start'] > 0) &  # Job actually started
                    ((df['time_end'] > submit_time) | (df['time_end'] == 0))  # Still running
                )
                
                # Jobs pending at submit time (within this batch only)
                pending_mask = (
                    same_partition &
                    (df['time_submit'] <= submit_time) &
                    ((df['time_start'] > submit_time) | (df['time_start'] == 0))
                )
                
                # Calculate metrics
                running_jobs = df[running_mask]
                df.loc[idx, 'jobs_running_at_submit'] = len(running_jobs)
                df.loc[idx, 'jobs_pending_at_submit'] = len(df[pending_mask])
                df.loc[idx, 'cores_allocated_at_submit'] = running_jobs['cpus_req'].sum()
        
        self.logger.warning("Fallback calculation complete - results may be less accurate")
        return df
    
    def load_and_combine_batches(self, batch_files: List[Path]) -> pd.DataFrame:
        """Load and combine multiple batch files using parallel processing."""
        self.logger.info(f"Loading {len(batch_files)} batch files in parallel...")
        
        # Determine number of workers
        preprocessing_config = self.config.get('preprocessing', {})
        n_jobs = preprocessing_config.get('n_jobs', -1)
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        # Process files in parallel
        processed_dfs = Parallel(n_jobs=n_jobs)(
            delayed(self.process_single_batch)(batch_file) 
            for batch_file in tqdm(batch_files, desc="Processing batch files")
        )
        
        # Filter out empty dataframes
        valid_dfs = [df for df in processed_dfs if not df.empty]
        
        if not valid_dfs:
            raise ValueError("No valid data found in batch files")
        
        # Combine all dataframes
        self.logger.info("Combining processed batches...")
        combined_df = pd.concat(valid_dfs, ignore_index=True)
        
        self.logger.info(f"Combined {len(combined_df)} total job records")
        return combined_df
    
    def save_processed_data(self, df: pd.DataFrame, output_filename: str):
        """Save processed data to file."""
        preprocessing_config = self.config.get('preprocessing', {})
        output_dir = Path(preprocessing_config.get('output_dir', 'processed_data'))
        output_dir.mkdir(exist_ok=True)
        
        # Determine output format from filename or use parquet
        if output_filename.endswith('.csv'):
            output_path = output_dir / output_filename
            df.to_csv(output_path, index=False)
        elif output_filename.endswith('.json'):
            output_path = output_dir / output_filename
            df.to_json(output_path, orient='records', date_format='iso')
        else:
            # Default to parquet
            if not output_filename.endswith('.parquet'):
                output_filename += '.parquet'
            output_path = output_dir / output_filename
            df.to_parquet(output_path, index=False)
        
        self.logger.info(f"Processed data saved to {output_path}")
        self.logger.info(f"Saved {len(df)} processed job records")
    
    def run_preprocessing(self, input_pattern: str, output_filename: str, max_files: Optional[int] = None):
        """Run the complete preprocessing pipeline."""
        try:
            self.logger.info("Starting SLURM data preprocessing")
            
                         # Find input files
             preprocessing_config = self.config.get('preprocessing', {})
             input_dir = Path(preprocessing_config.get('input_dir', 'data/raw'))
            
            # Get list of batch files matching pattern
            if '*' in input_pattern:
                batch_files = list(input_dir.glob(input_pattern))
            else:
                # Treat as exact filename
                batch_files = [input_dir / input_pattern]
                if not batch_files[0].exists():
                    # Try glob pattern
                    batch_files = list(input_dir.glob(f"*{input_pattern}*"))
            
            if not batch_files:
                raise ValueError(f"No batch files found matching pattern: {input_pattern}")
            
            # Limit number of files if specified
            if max_files:
                batch_files = batch_files[:max_files]
                self.logger.info(f"Limited to first {max_files} files")
            
            batch_files.sort()  # Process in order
            self.logger.info(f"Found {len(batch_files)} batch files to process")
            
            # Load and combine batch files
            combined_df = self.load_and_combine_batches(batch_files)
            
            # Calculate partition state metrics
            processed_df = self.calculate_partition_state_at_submit(combined_df)
            
            # Save processed data
            self.save_processed_data(processed_df, output_filename)
            
            # Print summary statistics
            self.print_summary_stats(processed_df)
            
            self.logger.info("Data preprocessing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def print_summary_stats(self, df: pd.DataFrame):
        """Print summary statistics of processed data."""
        self.logger.info("=== PREPROCESSING SUMMARY ===")
        self.logger.info(f"Total jobs processed: {len(df):,}")
        self.logger.info(f"Unique partitions: {df['partition'].nunique()}")
        self.logger.info(f"Unique accounts: {df['account'].nunique()}")
        self.logger.info(f"Date range: {pd.to_datetime(df['time_submit'], unit='s').min()} to {pd.to_datetime(df['time_submit'], unit='s').max()}")
        
        # Wait time statistics
        wait_stats = df['wait_time'].describe()
        self.logger.info(f"Wait time stats (seconds):")
        self.logger.info(f"  Mean: {wait_stats['mean']:.1f}")
        self.logger.info(f"  Median: {wait_stats['50%']:.1f}")
        self.logger.info(f"  Max: {wait_stats['max']:.1f}")
        
        # Resource request statistics
        self.logger.info(f"Resource requests:")
        self.logger.info(f"  CPUs: mean={df['cpus_req'].mean():.1f}, max={df['cpus_req'].max()}")
        self.logger.info(f"  Nodes: mean={df['nodes_req'].mean():.1f}, max={df['nodes_req'].max()}")
        self.logger.info(f"  GPUs: mean={df['gpus_req'].mean():.1f}, max={df['gpus_req'].max()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preprocess SLURM job data for ML training")
    parser.add_argument(
        "input_pattern",
        help="Pattern to match input batch files (e.g., 'slurm_jobs_batch_*.parquet' or 'batch_')"
    )
    parser.add_argument(
        "output_filename",
        help="Output filename for processed data"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of batch files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    try:
        preprocessor = SlurmDataPreprocessor(args.config)
        preprocessor.run_preprocessing(args.input_pattern, args.output_filename, args.max_files)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 