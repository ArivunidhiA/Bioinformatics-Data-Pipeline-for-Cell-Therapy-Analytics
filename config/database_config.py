"""
Database configuration for Cell Therapy Analytics Pipeline
Handles both SQLite (development) and PostgreSQL (production) connections
"""

import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration class with environment-specific settings"""
    
    def __init__(self, environment='development'):
        self.environment = environment
        self.base = declarative_base()
        self.metadata = MetaData()
        
    @property
    def database_url(self):
        """Get database URL based on environment"""
        if self.environment == 'production':
            return os.getenv('DATABASE_URL', 
                           'postgresql://biotech_user:secure_password_2024@localhost:5432/cell_therapy_db')
        else:
            # Development: Use SQLite
            return os.getenv('DATABASE_URL', 'sqlite:///./data/cell_therapy.db')
    
    def create_engine(self):
        """Create SQLAlchemy engine with appropriate configuration"""
        if self.environment == 'production':
            engine = create_engine(
                self.database_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
        else:
            # Development: SQLite with better performance
            engine = create_engine(
                self.database_url,
                connect_args={'check_same_thread': False},
                poolclass=StaticPool,
                echo=True
            )
        
        logger.info(f"Database engine created for {self.environment} environment")
        return engine
    
    def create_session_factory(self):
        """Create session factory for database operations"""
        engine = self.create_engine()
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal
    
    def get_session(self):
        """Get database session"""
        SessionLocal = self.create_session_factory()
        return SessionLocal()

# Global database configuration
db_config = DatabaseConfig(environment=os.getenv('ENVIRONMENT', 'development'))

# Database schemas and table definitions
CELL_THERAPY_SCHEMA = {
    'flow_cytometry_data': {
        'columns': [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'sample_id VARCHAR(50) NOT NULL',
            'batch_id VARCHAR(50) NOT NULL',
            'acquisition_date TIMESTAMP',
            'technician_id VARCHAR(50)',
            'protocol_version VARCHAR(20)',
            'total_events INTEGER',
            'live_cells_count INTEGER',
            'dead_cells_count INTEGER',
            'viability_percentage DECIMAL(5,2)',
            't_cells_count INTEGER',
            'nk_cells_count INTEGER',
            'b_cells_count INTEGER',
            'data_file_path VARCHAR(255)',
            'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        ],
        'indexes': [
            'CREATE INDEX idx_sample_id ON flow_cytometry_data(sample_id)',
            'CREATE INDEX idx_batch_id ON flow_cytometry_data(batch_id)',
            'CREATE INDEX idx_acquisition_date ON flow_cytometry_data(acquisition_date)'
        ]
    },
    
    'cell_counts': {
        'columns': [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'sample_id VARCHAR(50) NOT NULL',
            'batch_id VARCHAR(50) NOT NULL',
            'count_date TIMESTAMP',
            'total_cells INTEGER',
            'viable_cells INTEGER',
            'non_viable_cells INTEGER',
            'viability_percentage DECIMAL(5,2)',
            'cell_density DECIMAL(10,2)',
            'technician_id VARCHAR(50)',
            'counting_method VARCHAR(50)',
            'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        ],
        'indexes': [
            'CREATE INDEX idx_counts_sample_id ON cell_counts(sample_id)',
            'CREATE INDEX idx_counts_batch_id ON cell_counts(batch_id)'
        ]
    },
    
    'sample_metadata': {
        'columns': [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'sample_id VARCHAR(50) UNIQUE NOT NULL',
            'patient_id VARCHAR(50)',
            'cell_type VARCHAR(100)',
            'donor_id VARCHAR(50)',
            'collection_date DATE',
            'processing_date DATE',
            'storage_conditions VARCHAR(100)',
            'protocol_id VARCHAR(50)',
            'batch_id VARCHAR(50)',
            'quality_score DECIMAL(3,2)',
            'notes TEXT',
            'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        ],
        'indexes': [
            'CREATE INDEX idx_metadata_sample_id ON sample_metadata(sample_id)',
            'CREATE INDEX idx_metadata_patient_id ON sample_metadata(patient_id)',
            'CREATE INDEX idx_metadata_batch_id ON sample_metadata(batch_id)'
        ]
    },
    
    'processing_batches': {
        'columns': [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'batch_id VARCHAR(50) UNIQUE NOT NULL',
            'batch_name VARCHAR(100)',
            'start_date TIMESTAMP',
            'end_date TIMESTAMP',
            'status VARCHAR(20) DEFAULT "pending"',
            'total_samples INTEGER',
            'processed_samples INTEGER DEFAULT 0',
            'failed_samples INTEGER DEFAULT 0',
            'technician_id VARCHAR(50)',
            'protocol_version VARCHAR(20)',
            'quality_metrics JSON',
            'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        ],
        'indexes': [
            'CREATE INDEX idx_batches_batch_id ON processing_batches(batch_id)',
            'CREATE INDEX idx_batches_status ON processing_batches(status)'
        ]
    },
    
    'audit_log': {
        'columns': [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'user_id VARCHAR(50)',
            'action VARCHAR(100)',
            'table_name VARCHAR(100)',
            'record_id VARCHAR(50)',
            'old_values JSON',
            'new_values JSON',
            'change_reason TEXT',
            'approval_status VARCHAR(20) DEFAULT "pending"',
            'approved_by VARCHAR(50)',
            'approved_at TIMESTAMP'
        ],
        'indexes': [
            'CREATE INDEX idx_audit_timestamp ON audit_log(timestamp)',
            'CREATE INDEX idx_audit_user_id ON audit_log(user_id)',
            'CREATE INDEX idx_audit_table_name ON audit_log(table_name)'
        ]
    },
    
    'validation_rules': {
        'columns': [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'rule_name VARCHAR(100) UNIQUE NOT NULL',
            'rule_type VARCHAR(50)',
            'table_name VARCHAR(100)',
            'column_name VARCHAR(100)',
            'rule_definition JSON',
            'severity VARCHAR(20) DEFAULT "error"',
            'is_active BOOLEAN DEFAULT TRUE',
            'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        ],
        'indexes': [
            'CREATE INDEX idx_rules_rule_name ON validation_rules(rule_name)',
            'CREATE INDEX idx_rules_table_name ON validation_rules(table_name)'
        ]
    }
}

def initialize_database():
    """Initialize database with all tables and indexes"""
    engine = db_config.create_engine()
    
    # Create tables
    for table_name, table_def in CELL_THERAPY_SCHEMA.items():
        columns = ', '.join(table_def['columns'])
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        
        try:
            with engine.connect() as conn:
                conn.execute(create_table_sql)
                conn.commit()
                
                # Create indexes
                for index_sql in table_def.get('indexes', []):
                    try:
                        conn.execute(index_sql)
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"Index creation failed for {table_name}: {e}")
                        
            logger.info(f"Table {table_name} created successfully")
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
    
    logger.info("Database initialization completed")

if __name__ == "__main__":
    initialize_database() 