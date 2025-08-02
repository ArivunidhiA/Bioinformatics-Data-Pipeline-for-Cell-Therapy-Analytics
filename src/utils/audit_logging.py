"""
Audit Logging Module for Cell Therapy Analytics Pipeline
Provides comprehensive audit trail generation for all system changes
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuditLogging:
    """
    Audit Logging System for Cell Therapy Analytics Pipeline
    
    Implements:
    - Comprehensive audit trail generation
    - Database and file-based logging
    - Change tracking for all system modifications
    - Compliance reporting
    - Data integrity verification
    """
    
    def __init__(self, db_path: str = "data/audit_log.db"):
        """Initialize the Audit Logging system"""
        self.db_path = db_path
        self.audit_db = None
        
        # Ensure audit database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize audit database
        self._initialize_audit_database()
        
        # Audit logging statistics
        self.stats = {
            'total_logs': 0,
            'logs_today': 0,
            'compliance_score': 100.0,
            'data_integrity_verified': True
        }
        
        logger.info("Audit Logging system initialized")
    
    def _initialize_audit_database(self):
        """Initialize SQLite audit database"""
        try:
            self.audit_db = sqlite3.connect(self.db_path)
            
            # Create audit log table
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id VARCHAR(50) NOT NULL,
                action VARCHAR(100) NOT NULL,
                table_name VARCHAR(100),
                record_id VARCHAR(50),
                old_values TEXT,
                new_values TEXT,
                change_reason TEXT,
                approval_status VARCHAR(20) DEFAULT 'pending',
                approved_by VARCHAR(50),
                approved_at TIMESTAMP,
                session_id VARCHAR(100),
                ip_address VARCHAR(45),
                user_agent TEXT,
                severity VARCHAR(20) DEFAULT 'info',
                hash_value VARCHAR(64)
            )
            """
            
            self.audit_db.execute(create_table_sql)
            
            # Create indexes for better performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_log(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)",
                "CREATE INDEX IF NOT EXISTS idx_audit_table_name ON audit_log(table_name)",
                "CREATE INDEX IF NOT EXISTS idx_audit_record_id ON audit_log(record_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_approval_status ON audit_log(approval_status)"
            ]
            
            for index_sql in indexes:
                self.audit_db.execute(index_sql)
            
            self.audit_db.commit()
            logger.info(f"Audit database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing audit database: {e}")
            self.audit_db = None
    
    def log_action(self, user_id: str, action: str, table_name: Optional[str] = None,
                  record_id: Optional[str] = None, old_values: Optional[Dict[str, Any]] = None,
                  new_values: Optional[Dict[str, Any]] = None, change_reason: Optional[str] = None,
                  approval_status: str = 'pending', approved_by: Optional[str] = None,
                  session_id: Optional[str] = None, ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None, severity: str = 'info') -> bool:
        """
        Log an action to the audit trail
        
        Args:
            user_id: User performing the action
            action: Action being performed
            table_name: Database table affected
            record_id: Record identifier
            old_values: Previous values (for updates/deletes)
            new_values: New values (for creates/updates)
            change_reason: Reason for the change
            approval_status: Approval status
            approved_by: User who approved
            session_id: Session identifier
            ip_address: IP address of user
            user_agent: User agent string
            severity: Log severity (info, warning, error, critical)
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            # Generate hash for data integrity
            hash_value = self._generate_audit_hash(user_id, action, table_name, record_id, 
                                                 old_values, new_values, change_reason)
            
            # Prepare values for database storage
            old_values_json = json.dumps(old_values) if old_values else None
            new_values_json = json.dumps(new_values) if new_values else None
            
            # Insert into audit database
            insert_sql = """
            INSERT INTO audit_log (
                user_id, action, table_name, record_id, old_values, new_values,
                change_reason, approval_status, approved_by, session_id, ip_address,
                user_agent, severity, hash_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                user_id, action, table_name, record_id, old_values_json, new_values_json,
                change_reason, approval_status, approved_by, session_id, ip_address,
                user_agent, severity, hash_value
            )
            
            if self.audit_db:
                self.audit_db.execute(insert_sql, params)
                self.audit_db.commit()
            
            # Also write to file-based log for redundancy
            self._write_to_file_log(user_id, action, table_name, record_id, old_values, 
                                  new_values, change_reason, severity, hash_value)
            
            # Update statistics
            self._update_audit_statistics()
            
            logger.debug(f"Audit log entry created: {action} by {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging audit action: {e}")
            return False
    
    def get_audit_trail(self, user_id: Optional[str] = None, action: Optional[str] = None,
                       table_name: Optional[str] = None, record_id: Optional[str] = None,
                       start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve audit trail entries
        
        Args:
            user_id: Filter by user ID
            action: Filter by action
            table_name: Filter by table name
            record_id: Filter by record ID
            start_date: Start date for filtering
            end_date: End date for filtering
            limit: Maximum number of records to return
            
        Returns:
            List of audit log entries
        """
        try:
            if not self.audit_db:
                return []
            
            # Build query
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if action:
                query += " AND action = ?"
                params.append(action)
            
            if table_name:
                query += " AND table_name = ?"
                params.append(table_name)
            
            if record_id:
                query += " AND record_id = ?"
                params.append(record_id)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            cursor = self.audit_db.execute(query, params)
            columns = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries
            audit_entries = []
            for row in cursor.fetchall():
                entry = dict(zip(columns, row))
                
                # Parse JSON values
                if entry.get('old_values'):
                    entry['old_values'] = json.loads(entry['old_values'])
                if entry.get('new_values'):
                    entry['new_values'] = json.loads(entry['new_values'])
                
                audit_entries.append(entry)
            
            return audit_entries
            
        except Exception as e:
            logger.error(f"Error retrieving audit trail: {e}")
            return []
    
    def get_change_history(self, table_name: str, record_id: str) -> List[Dict[str, Any]]:
        """
        Get change history for a specific record
        
        Args:
            table_name: Database table name
            record_id: Record identifier
            
        Returns:
            List of changes for the record
        """
        try:
            return self.get_audit_trail(table_name=table_name, record_id=record_id)
        except Exception as e:
            logger.error(f"Error getting change history: {e}")
            return []
    
    def verify_data_integrity(self, start_date: Optional[datetime] = None, 
                            end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Verify data integrity of audit trail
        
        Args:
            start_date: Start date for verification
            end_date: End date for verification
            
        Returns:
            Data integrity verification results
        """
        try:
            if not self.audit_db:
                return {'verified': False, 'error': 'Audit database not available'}
            
            # Get audit entries for verification period
            audit_entries = self.get_audit_trail(start_date=start_date, end_date=end_date, limit=10000)
            
            integrity_results = {
                'verified': True,
                'total_entries': len(audit_entries),
                'verified_entries': 0,
                'failed_entries': 0,
                'errors': []
            }
            
            for entry in audit_entries:
                # Recalculate hash
                calculated_hash = self._generate_audit_hash(
                    entry['user_id'], entry['action'], entry['table_name'], entry['record_id'],
                    entry.get('old_values'), entry.get('new_values'), entry.get('change_reason')
                )
                
                # Compare with stored hash
                if entry.get('hash_value') == calculated_hash:
                    integrity_results['verified_entries'] += 1
                else:
                    integrity_results['failed_entries'] += 1
                    integrity_results['errors'].append({
                        'id': entry['id'],
                        'timestamp': entry['timestamp'],
                        'expected_hash': calculated_hash,
                        'stored_hash': entry.get('hash_value')
                    })
            
            # Update overall verification status
            if integrity_results['failed_entries'] > 0:
                integrity_results['verified'] = False
                self.stats['data_integrity_verified'] = False
            else:
                self.stats['data_integrity_verified'] = True
            
            return integrity_results
            
        except Exception as e:
            logger.error(f"Error verifying data integrity: {e}")
            return {'verified': False, 'error': str(e)}
    
    def generate_compliance_report(self, start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate compliance report for audit trail
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            Compliance report
        """
        try:
            if not self.audit_db:
                return {'error': 'Audit database not available'}
            
            # Get audit entries for report period
            audit_entries = self.get_audit_trail(start_date=start_date, end_date=end_date, limit=10000)
            
            # Calculate compliance metrics
            total_actions = len(audit_entries)
            approved_actions = len([e for e in audit_entries if e.get('approval_status') == 'approved'])
            pending_actions = len([e for e in audit_entries if e.get('approval_status') == 'pending'])
            rejected_actions = len([e for e in audit_entries if e.get('approval_status') == 'rejected'])
            
            # Calculate compliance score
            compliance_score = (approved_actions / max(1, total_actions)) * 100
            
            # Group by action type
            action_counts = {}
            for entry in audit_entries:
                action = entry['action']
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Group by user
            user_counts = {}
            for entry in audit_entries:
                user = entry['user_id']
                user_counts[user] = user_counts.get(user, 0) + 1
            
            # Generate report
            report = {
                'report_period': {
                    'start_date': start_date.isoformat() if start_date else 'All time',
                    'end_date': end_date.isoformat() if end_date else 'All time'
                },
                'summary': {
                    'total_actions': total_actions,
                    'approved_actions': approved_actions,
                    'pending_actions': pending_actions,
                    'rejected_actions': rejected_actions,
                    'compliance_score': round(compliance_score, 2)
                },
                'action_breakdown': action_counts,
                'user_breakdown': user_counts,
                'data_integrity': self.verify_data_integrity(start_date, end_date),
                'recommendations': self._generate_compliance_recommendations(audit_entries)
            }
            
            # Update statistics
            self.stats['compliance_score'] = compliance_score
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {'error': str(e)}
    
    def _generate_audit_hash(self, user_id: str, action: str, table_name: Optional[str],
                           record_id: Optional[str], old_values: Optional[Dict[str, Any]],
                           new_values: Optional[Dict[str, Any]], change_reason: Optional[str]) -> str:
        """Generate hash for audit trail integrity"""
        try:
            # Create content string for hashing
            content = f"{user_id}|{action}|{table_name or ''}|{record_id or ''}|"
            content += f"{json.dumps(old_values, sort_keys=True) if old_values else ''}|"
            content += f"{json.dumps(new_values, sort_keys=True) if new_values else ''}|"
            content += f"{change_reason or ''}"
            
            # Generate SHA-256 hash
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating audit hash: {e}")
            return ""
    
    def _write_to_file_log(self, user_id: str, action: str, table_name: Optional[str],
                          record_id: Optional[str], old_values: Optional[Dict[str, Any]],
                          new_values: Optional[Dict[str, Any]], change_reason: Optional[str],
                          severity: str, hash_value: str):
        """Write audit log to file for redundancy"""
        try:
            log_dir = Path("data/audit_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create daily log file
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = log_dir / f"audit_log_{today}.jsonl"
            
            # Create log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'action': action,
                'table_name': table_name,
                'record_id': record_id,
                'old_values': old_values,
                'new_values': new_values,
                'change_reason': change_reason,
                'severity': severity,
                'hash_value': hash_value
            }
            
            # Write to file
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing to file log: {e}")
    
    def _update_audit_statistics(self):
        """Update audit logging statistics"""
        try:
            self.stats['total_logs'] += 1
            
            # Count today's logs
            today = datetime.now().date()
            today_logs = self.get_audit_trail(
                start_date=datetime.combine(today, datetime.min.time()),
                end_date=datetime.combine(today, datetime.max.time())
            )
            self.stats['logs_today'] = len(today_logs)
            
        except Exception as e:
            logger.error(f"Error updating audit statistics: {e}")
    
    def _generate_compliance_recommendations(self, audit_entries: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations based on audit trail"""
        recommendations = []
        
        try:
            # Check for high number of pending approvals
            pending_count = len([e for e in audit_entries if e.get('approval_status') == 'pending'])
            if pending_count > 10:
                recommendations.append("High number of pending approvals. Consider streamlining approval process.")
            
            # Check for rejected changes
            rejected_count = len([e for e in audit_entries if e.get('approval_status') == 'rejected'])
            if rejected_count > 5:
                recommendations.append("Multiple rejected changes detected. Review change request procedures.")
            
            # Check for data integrity issues
            integrity_check = self.verify_data_integrity()
            if not integrity_check.get('verified', True):
                recommendations.append("Data integrity issues detected. Investigate audit trail integrity.")
            
            # Check for unusual activity patterns
            user_activity = {}
            for entry in audit_entries:
                user = entry['user_id']
                user_activity[user] = user_activity.get(user, 0) + 1
            
            high_activity_users = [user for user, count in user_activity.items() if count > 50]
            if high_activity_users:
                recommendations.append(f"High activity detected for users: {', '.join(high_activity_users)}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating compliance recommendations: {e}")
            return ["Error generating recommendations"]
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        return {
            'total_logs': self.stats['total_logs'],
            'logs_today': self.stats['logs_today'],
            'compliance_score_percentage': self.stats['compliance_score'],
            'data_integrity_verified': self.stats['data_integrity_verified'],
            'audit_database_available': self.audit_db is not None
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 365):
        """
        Clean up old audit logs
        
        Args:
            days_to_keep: Number of days to keep logs
        """
        try:
            if not self.audit_db:
                return
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Delete old entries from database
            delete_sql = "DELETE FROM audit_log WHERE timestamp < ?"
            self.audit_db.execute(delete_sql, (cutoff_date.isoformat(),))
            self.audit_db.commit()
            
            # Clean up old file logs
            log_dir = Path("data/audit_logs")
            if log_dir.exists():
                for log_file in log_dir.glob("audit_log_*.jsonl"):
                    try:
                        file_date_str = log_file.stem.split('_')[-1]
                        file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()
                        if file_date < cutoff_date.date():
                            log_file.unlink()
                            logger.info(f"Deleted old audit log file: {log_file}")
                    except Exception as e:
                        logger.warning(f"Error processing log file {log_file}: {e}")
            
            logger.info(f"Cleaned up audit logs older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if self.audit_db:
                self.audit_db.close()
        except Exception as e:
            logger.error(f"Error closing audit database: {e}") 