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
    
    def extract_job_data(self) -> pd.DataFrame:
        """Extract job data from SLURM database."""
        if not self.engine:
            raise RuntimeError("Database connection not established. Call connect_to_database() first.")
        
        query = self.get_job_data_query()
        batch_size = self.config.get('extraction', {}).get('batch_size', 10000)
        
        self.logger.info("Starting job data extraction...")
        
        try:
            # For large datasets, we might want to use chunking
            chunks = []
            offset = 0
            
            while True:
                chunked_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
                self.logger.debug(f"Executing query with offset {offset}")
                
                chunk = pd.read_sql(chunked_query, self.engine)
                
                if chunk.empty:
                    break
                
                chunks.append(chunk)
                offset += batch_size
                
                self.logger.info(f"Extracted {len(chunk)} records (total: {offset})")
            
            if not chunks:
                self.logger.warning("No job data found")
                return pd.DataFrame()
            
            df = pd.concat(chunks, ignore_index=True)
            self.logger.info(f"Successfully extracted {len(df)} total job records")
            
            return df
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error extracting job data: {e}")
            raise
    

    
    def save_data(self, df: pd.DataFrame, filename: Optional[str] = None):
        """Save extracted data to file."""
        if df.empty:
            self.logger.warning("No data to save")
            return
        
        extraction_config = self.config.get('extraction', {})
        output_dir = Path(extraction_config.get('output_dir', 'data'))
        output_format = extraction_config.get('output_format', 'parquet')
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"slurm_jobs_{timestamp}.{output_format}"
        
        output_path = output_dir / filename
        
        try:
            if output_format == 'csv':
                df.to_csv(output_path, index=False)
            elif output_format == 'parquet':
                df.to_parquet(output_path, index=False)
            elif output_format == 'json':
                df.to_json(output_path, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            self.logger.info(f"Data saved to {output_path}")
            self.logger.info(f"Saved {len(df)} records in {output_format} format")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise
    
    def run_extraction(self, output_filename: Optional[str] = None):
        """Run the complete data extraction process."""
        try:
            self.logger.info("Starting SLURM data extraction process")
            
            # Connect to database
            self.connect_to_database()
            
            # Extract data
            df = self.extract_job_data()
            
            if df.empty:
                self.logger.warning("No data extracted")
                return
            
            # Save raw data
            self.save_data(df, output_filename)
            
            self.logger.info("Data extraction process completed successfully")
            
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
    
    args = parser.parse_args()
    
    try:
        extractor = SlurmDataExtractor(args.config)
        
        if args.dry_run:
            extractor.connect_to_database()
            print("✓ Database connection successful")
        else:
            extractor.run_extraction(args.output)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 