#!/usr/bin/env python3
"""
SLURM Data Extractor

This script connects to a SLURM accounting database and extracts job data
for use in training predictive AI models.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import quote_plus

import yaml
import pandas as pd
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class SlurmDataExtractor:
    """Extract job data from SLURM accounting database."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the extractor with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.engine = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create logs directory if needed
        if log_config.get('log_to_file', False):
            log_file = log_config.get('log_file', 'logs/slurm_extractor.log')
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
    
    def connect_to_database(self):
        """Establish connection to SLURM database."""
        db_config = self.config['database']
        
        # URL-encode username and password to handle special characters like @, :, etc.
        username = quote_plus(str(db_config['username']))
        password = quote_plus(str(db_config['password']))
        host = db_config['host']
        port = db_config['port']
        database = db_config['database']
        
        connection_string = (
            f"mysql+pymysql://{username}:{password}"
            f"@{host}:{port}/{database}"
        )
        
        try:
            self.engine = create_engine(connection_string)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.logger.info("Successfully connected to SLURM database")
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def get_job_data_query(self) -> str:
        """Build SQL query to extract job data."""
        # Get cluster name from config to build correct table name
        db_config = self.config['database']
        cluster_name = db_config.get('cluster_name', 'cluster')
        job_table = f"{cluster_name}_job_table"
        
        # Main job data query
        base_query = f"""
        SELECT 
            job_db_inx,
            id_job,
            job_name,
            `partition`,
            account,
            id_user,
            id_group,
            state,
            exit_code,
            priority,
            time_submit,
            time_eligible,
            time_start,
            time_end,
            time_suspended,
            timelimit,
            cpus_req,
            mem_req,
            nodes_alloc,
            nodelist,
            work_dir,
            tres_alloc,
            tres_req,
            id_assoc,
            id_qos,
            array_task_str,
            array_max_tasks,
            constraints,
            derived_ec,
            derived_es,
            flags,
            het_job_id,
            het_job_offset,
            wckey,
            std_err,
            std_in,
            std_out,
            submit_line
        FROM {job_table}
        WHERE deleted = 0
        """
        
        # Add date filtering if specified
        extraction_config = self.config.get('extraction', {})
        conditions = []
        
        if extraction_config.get('start_date'):
            conditions.append(f"time_submit >= UNIX_TIMESTAMP('{extraction_config['start_date']}')")
        
        if extraction_config.get('end_date'):
            conditions.append(f"time_submit <= UNIX_TIMESTAMP('{extraction_config['end_date']}')")
        
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        base_query += " ORDER BY time_submit"
        
        return base_query
    
    def get_total_record_count(self) -> int:
        """Get total number of records to be extracted."""
        if not self.engine:
            raise RuntimeError("Database connection not established.")
        
        db_config = self.config['database']
        cluster_name = db_config.get('cluster_name', 'cluster')
        job_table = f"{cluster_name}_job_table"
        
        count_query = f"SELECT COUNT(*) as total FROM {job_table} WHERE deleted = 0"
        
        # Add date filtering if specified
        extraction_config = self.config.get('extraction', {})
        conditions = []
        
        if extraction_config.get('start_date'):
            conditions.append(f"time_submit >= UNIX_TIMESTAMP('{extraction_config['start_date']}')")
        
        if extraction_config.get('end_date'):
            conditions.append(f"time_submit <= UNIX_TIMESTAMP('{extraction_config['end_date']}')")
        
        if conditions:
            count_query += " AND " + " AND ".join(conditions)
        
        try:
            result = pd.read_sql(count_query, self.engine)
            return int(result.iloc[0]['total'])
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting record count: {e}")
            raise

    def load_checkpoint(self, checkpoint_file: Path) -> Dict[str, Any]:
        """Load extraction checkpoint from file."""
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
        return {'offset': 0, 'total_extracted': 0, 'batch_files': []}

    def save_checkpoint(self, checkpoint_file: Path, offset: int, total_extracted: int, batch_files: list):
        """Save extraction checkpoint to file."""
        checkpoint_data = {
            'offset': offset,
            'total_extracted': total_extracted,
            'batch_files': batch_files,
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except IOError as e:
            self.logger.warning(f"Could not save checkpoint: {e}")

    def extract_job_data_chunked(self, output_filename: Optional[str] = None, checkpoint_file: Optional[str] = None) -> list:
        """Extract job data in checkpointable chunks"""
        if not self.engine:
            raise RuntimeError("Database connection not established. Call connect_to_database() first.")
        
        # Setup parameters
        extraction_config = self.config.get('extraction', {})
        batch_size = extraction_config.get('batch_size', 5000) # default to batch size of 5000
        output_dir = Path(extraction_config.get('output_dir', 'data'))
        output_format = extraction_config.get('output_format', 'parquet')
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Setup checkpoint file
        base_filename = output_filename or "slurm_jobs"
        if checkpoint_file:
            checkpoint_path = Path(checkpoint_file)
        else:
            checkpoint_path = output_dir / f"{base_filename}_checkpoint.json"
        
        # Load checkpoint if exists
        checkpoint = self.load_checkpoint(checkpoint_path)
        start_offset = checkpoint['offset']
        total_extracted = checkpoint['total_extracted']
        batch_files = checkpoint['batch_files']
        
        if start_offset > 0:
            self.logger.info(f"Resuming extraction from offset {start_offset} ({total_extracted} records already extracted)")
            self.logger.info(f"Using checkpoint file: {checkpoint_path}")
        
        # Get total record count
        total_records = self.get_total_record_count()
        self.logger.info(f"Total records to extract: {total_records:,}")
        
        query = self.get_job_data_query()
        
        try:
            offset = start_offset
            batch_number = len(batch_files)
            
            while offset < total_records:
                chunked_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
                self.logger.info(f"Extracting batch {batch_number + 1}: offset {offset:,} (Progress: {(offset/total_records)*100:.1f}%)")
                
                chunk = pd.read_sql(chunked_query, self.engine)
                
                if chunk.empty:
                    self.logger.info("No more data to extract")
                    break
                
                # Save this batch to file
                batch_filename = f"{base_filename}_batch_{batch_number:04d}.{output_format}"
                batch_path = output_dir / batch_filename
                
                if output_format == 'csv':
                    chunk.to_csv(batch_path, index=False)
                elif output_format == 'parquet':
                    chunk.to_parquet(batch_path, index=False)
                elif output_format == 'json':
                    chunk.to_json(batch_path, orient='records', date_format='iso')
                
                batch_files.append(str(batch_filename))
                records_in_batch = len(chunk)
                total_extracted += records_in_batch
                offset += batch_size
                batch_number += 1
                
                self.logger.info(f"Saved batch {batch_number} to {batch_filename} ({records_in_batch:,} records)")
                
                # Save checkpoint after each batch
                self.save_checkpoint(checkpoint_path, offset, total_extracted, batch_files)
                
                # Clear chunk from memory
                del chunk
            
            self.logger.info(f"Extraction completed! Total records extracted: {total_extracted:,}")
            self.logger.info(f"Created {len(batch_files)} batch files")
            
            # Clean up checkpoint file on successful completion
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                self.logger.info("Removed checkpoint file (extraction completed successfully)")
            
            return batch_files
            
        except Exception as e:
            self.logger.error(f"Error during chunked extraction: {e}")
            self.logger.info(f"Checkpoint saved. Resume with same command to continue from offset {offset}")
            raise
    

    
    def consolidate_batch_files(self, batch_files: list, final_filename: Optional[str] = None):
        """Consolidate multiple batch files into a single file (optional)."""
        if not batch_files:
            self.logger.warning("No batch files to consolidate")
            return
        
        extraction_config = self.config.get('extraction', {})
        output_dir = Path(extraction_config.get('output_dir', 'data'))
        output_format = extraction_config.get('output_format', 'parquet')
        
        if not final_filename:
            final_filename = f"slurm_jobs_consolidated.{output_format}"
        
        final_path = output_dir / final_filename
        
        self.logger.info(f"Consolidating {len(batch_files)} batch files into {final_filename}")
        
        try:
            # Read and concatenate all batch files
            chunks = []
            for batch_file in batch_files:
                batch_path = output_dir / batch_file
                if output_format == 'csv':
                    chunk = pd.read_csv(batch_path)
                elif output_format == 'parquet':
                    chunk = pd.read_parquet(batch_path)
                elif output_format == 'json':
                    chunk = pd.read_json(batch_path, orient='records')
                chunks.append(chunk)
            
            # Concatenate all chunks
            df = pd.concat(chunks, ignore_index=True)
            
            # Save consolidated file
            if output_format == 'csv':
                df.to_csv(final_path, index=False)
            elif output_format == 'parquet':
                df.to_parquet(final_path, index=False)
            elif output_format == 'json':
                df.to_json(final_path, orient='records', date_format='iso')
            
            self.logger.info(f"Consolidated file saved: {final_path} ({len(df):,} total records)")
            
            # Optionally remove batch files
            cleanup = extraction_config.get('cleanup_batch_files', False)
            if cleanup:
                for batch_file in batch_files:
                    batch_path = output_dir / batch_file
                    if batch_path.exists():
                        batch_path.unlink()
                self.logger.info("Cleaned up individual batch files")
            
        except Exception as e:
            self.logger.error(f"Error consolidating batch files: {e}")
            raise
    
    def run_extraction(self, output_filename: Optional[str] = None, consolidate: bool = False, checkpoint_file: Optional[str] = None):
        """Run the complete data extraction process using chunked extraction."""
        try:
            self.logger.info("Starting SLURM chunked data extraction process")
            
            # Connect to database
            self.connect_to_database()
            
            # Extract data in chunks
            batch_files = self.extract_job_data_chunked(output_filename, checkpoint_file)
            
            if not batch_files:
                self.logger.warning("No data extracted")
                return
            
            # Optionally consolidate batch files
            if consolidate:
                self.consolidate_batch_files(batch_files, output_filename)
            
            self.logger.info("Data extraction process completed successfully")
            self.logger.info(f"Output files: {len(batch_files)} batch files in data/ directory")
            
        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            raise
        finally:
            if self.engine:
                self.engine.dispose()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract SLURM job data for ML training")
    parser.add_argument(
        "--config", 
        default="config.yaml", 
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--output", 
        help="Output filename (optional, auto-generated if not provided)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Test database connection without extracting data"
    )
    parser.add_argument(
        "--consolidate", 
        action="store_true", 
        help="Consolidate batch files into single file after extraction"
    )
    parser.add_argument(
        "--checkpoint", 
        help="Path to checkpoint file for resuming extraction"
    )
    
    args = parser.parse_args()
    
    try:
        extractor = SlurmDataExtractor(args.config)
        
        if args.dry_run:
            extractor.connect_to_database()
            total_records = extractor.get_total_record_count()
            print("Database connection successful")
            print(f"Total records found: {total_records:,}")
        else:
            extractor.run_extraction(args.output, args.consolidate, args.checkpoint)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 