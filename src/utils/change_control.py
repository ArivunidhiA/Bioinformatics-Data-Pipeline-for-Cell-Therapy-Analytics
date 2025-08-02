"""
Change Control Module for Cell Therapy Analytics Pipeline
Implements version control, approval workflows, and change management
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import git
from dataclasses import dataclass, asdict

from .audit_logging import AuditLogging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChangeRequest:
    """Data class for change request information"""
    change_id: str
    user_id: str
    change_type: str
    description: str
    affected_components: List[str]
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    priority: str  # low, medium, high, critical
    impact_assessment: str
    created_at: datetime
    status: str  # pending, approved, rejected, implemented
    approvals: List[Dict[str, Any]]
    implementation_date: Optional[datetime] = None
    rollback_plan: Optional[str] = None

class ChangeControl:
    """
    Change Control System for Cell Therapy Analytics Pipeline
    
    Implements:
    - Version control for analysis parameters
    - Approval workflows for system changes
    - Audit trail generation
    - Change impact assessment
    - Rollback capabilities
    """
    
    def __init__(self, repository_path: str = ".", config_path: str = "config/validation_rules.json"):
        """Initialize the Change Control system"""
        self.repository_path = Path(repository_path)
        self.config_path = config_path
        self.audit_logger = AuditLogging()
        
        # Change control statistics
        self.stats = {
            'total_changes': 0,
            'approved_changes': 0,
            'rejected_changes': 0,
            'pending_changes': 0,
            'change_control_efficiency': 100.0  # Target efficiency
        }
        
        # Initialize Git repository if not exists
        self._initialize_git_repository()
        
        # Load change control configuration
        self.change_config = self._load_change_config()
        
        logger.info("Change Control system initialized")
    
    def _initialize_git_repository(self):
        """Initialize Git repository for version control"""
        try:
            if not (self.repository_path / '.git').exists():
                repo = git.Repo.init(self.repository_path)
                logger.info(f"Initialized Git repository at {self.repository_path}")
            else:
                repo = git.Repo(self.repository_path)
                logger.info(f"Using existing Git repository at {self.repository_path}")
            
            self.repo = repo
            
        except Exception as e:
            logger.error(f"Error initializing Git repository: {e}")
            self.repo = None
    
    def _load_change_config(self) -> Dict[str, Any]:
        """Load change control configuration"""
        try:
            with open(self.config_path, 'r') as file:
                config = json.load(file)
            
            # Extract change control rules
            change_rules = config.get('validation_rules', {}).get('change_control', {})
            
            return {
                'require_approval': change_rules.get('require_approval', True),
                'approval_threshold': change_rules.get('approval_threshold', 2),
                'change_logging': change_rules.get('change_logging', True),
                'rollback_enabled': change_rules.get('rollback_enabled', True),
                'approval_workflow': {
                    'low_priority': ['team_lead'],
                    'medium_priority': ['team_lead', 'manager'],
                    'high_priority': ['team_lead', 'manager', 'director'],
                    'critical_priority': ['team_lead', 'manager', 'director', 'cto']
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading change control configuration: {e}")
            return self._get_default_change_config()
    
    def _get_default_change_config(self) -> Dict[str, Any]:
        """Get default change control configuration"""
        return {
            'require_approval': True,
            'approval_threshold': 2,
            'change_logging': True,
            'rollback_enabled': True,
            'approval_workflow': {
                'low_priority': ['team_lead'],
                'medium_priority': ['team_lead', 'manager'],
                'high_priority': ['team_lead', 'manager', 'director'],
                'critical_priority': ['team_lead', 'manager', 'director', 'cto']
            }
        }
    
    def create_change_request(self, user_id: str, change_type: str, description: str,
                            affected_components: List[str], old_values: Dict[str, Any],
                            new_values: Dict[str, Any], priority: str = 'medium') -> ChangeRequest:
        """
        Create a new change request
        
        Args:
            user_id: User requesting the change
            change_type: Type of change (config, code, data, process)
            description: Description of the change
            affected_components: List of affected system components
            old_values: Previous values
            new_values: New values
            priority: Change priority (low, medium, high, critical)
            
        Returns:
            ChangeRequest object
        """
        try:
            # Generate unique change ID
            change_id = self._generate_change_id(user_id, change_type, description)
            
            # Assess change impact
            impact_assessment = self._assess_change_impact(change_type, affected_components, new_values)
            
            # Create change request
            change_request = ChangeRequest(
                change_id=change_id,
                user_id=user_id,
                change_type=change_type,
                description=description,
                affected_components=affected_components,
                old_values=old_values,
                new_values=new_values,
                priority=priority,
                impact_assessment=impact_assessment,
                created_at=datetime.now(),
                status='pending',
                approvals=[],
                rollback_plan=self._generate_rollback_plan(old_values, new_values)
            )
            
            # Log change request creation
            self.audit_logger.log_action(
                user_id=user_id,
                action="change_request_created",
                table_name="change_control",
                record_id=change_id,
                old_values=old_values,
                new_values=new_values,
                change_reason=description
            )
            
            # Save change request
            self._save_change_request(change_request)
            
            # Update statistics
            self.stats['total_changes'] += 1
            self.stats['pending_changes'] += 1
            
            logger.info(f"Change request created: {change_id} by {user_id}")
            return change_request
            
        except Exception as e:
            logger.error(f"Error creating change request: {e}")
            raise
    
    def approve_change_request(self, change_id: str, approver_id: str, 
                             approval_notes: str = "") -> bool:
        """
        Approve a change request
        
        Args:
            change_id: Change request ID
            approver_id: User approving the change
            approval_notes: Notes from the approver
            
        Returns:
            True if approved, False otherwise
        """
        try:
            # Load change request
            change_request = self._load_change_request(change_id)
            if not change_request:
                raise ValueError(f"Change request {change_id} not found")
            
            # Check if user can approve this change
            if not self._can_approve_change(approver_id, change_request):
                logger.warning(f"User {approver_id} cannot approve change {change_id}")
                return False
            
            # Add approval
            approval = {
                'approver_id': approver_id,
                'approval_notes': approval_notes,
                'approval_date': datetime.now(),
                'approval_status': 'approved'
            }
            
            change_request.approvals.append(approval)
            
            # Check if enough approvals
            required_approvals = self._get_required_approvals(change_request.priority)
            if len(change_request.approvals) >= required_approvals:
                change_request.status = 'approved'
                self.stats['approved_changes'] += 1
                self.stats['pending_changes'] -= 1
            
            # Save updated change request
            self._save_change_request(change_request)
            
            # Log approval
            self.audit_logger.log_action(
                user_id=approver_id,
                action="change_request_approved",
                table_name="change_control",
                record_id=change_id,
                change_reason=f"Approved by {approver_id}: {approval_notes}"
            )
            
            logger.info(f"Change request {change_id} approved by {approver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error approving change request {change_id}: {e}")
            return False
    
    def reject_change_request(self, change_id: str, rejector_id: str, 
                            rejection_reason: str) -> bool:
        """
        Reject a change request
        
        Args:
            change_id: Change request ID
            rejector_id: User rejecting the change
            rejection_reason: Reason for rejection
            
        Returns:
            True if rejected, False otherwise
        """
        try:
            # Load change request
            change_request = self._load_change_request(change_id)
            if not change_request:
                raise ValueError(f"Change request {change_id} not found")
            
            # Update status
            change_request.status = 'rejected'
            self.stats['rejected_changes'] += 1
            self.stats['pending_changes'] -= 1
            
            # Save updated change request
            self._save_change_request(change_request)
            
            # Log rejection
            self.audit_logger.log_action(
                user_id=rejector_id,
                action="change_request_rejected",
                table_name="change_control",
                record_id=change_id,
                change_reason=f"Rejected by {rejector_id}: {rejection_reason}"
            )
            
            logger.info(f"Change request {change_id} rejected by {rejector_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rejecting change request {change_id}: {e}")
            return False
    
    def implement_change(self, change_id: str, implementer_id: str) -> bool:
        """
        Implement an approved change
        
        Args:
            change_id: Change request ID
            implementer_id: User implementing the change
            
        Returns:
            True if implemented, False otherwise
        """
        try:
            # Load change request
            change_request = self._load_change_request(change_id)
            if not change_request:
                raise ValueError(f"Change request {change_id} not found")
            
            if change_request.status != 'approved':
                raise ValueError(f"Change request {change_id} is not approved")
            
            # Create Git commit for the change
            commit_message = f"Implement change {change_id}: {change_request.description}"
            
            # Apply changes to files
            self._apply_changes_to_files(change_request)
            
            # Commit changes to Git
            if self.repo:
                self.repo.index.add('*')
                self.repo.index.commit(commit_message)
                logger.info(f"Changes committed to Git: {commit_message}")
            
            # Update change request status
            change_request.status = 'implemented'
            change_request.implementation_date = datetime.now()
            
            # Save updated change request
            self._save_change_request(change_request)
            
            # Log implementation
            self.audit_logger.log_action(
                user_id=implementer_id,
                action="change_implemented",
                table_name="change_control",
                record_id=change_id,
                change_reason=f"Implemented by {implementer_id}"
            )
            
            logger.info(f"Change {change_id} implemented by {implementer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error implementing change {change_id}: {e}")
            return False
    
    def rollback_change(self, change_id: str, rollback_user_id: str, 
                       rollback_reason: str) -> bool:
        """
        Rollback an implemented change
        
        Args:
            change_id: Change request ID
            rollback_user_id: User performing the rollback
            rollback_reason: Reason for rollback
            
        Returns:
            True if rolled back, False otherwise
        """
        try:
            # Load change request
            change_request = self._load_change_request(change_id)
            if not change_request:
                raise ValueError(f"Change request {change_id} not found")
            
            if change_request.status != 'implemented':
                raise ValueError(f"Change request {change_id} is not implemented")
            
            # Apply rollback using Git
            if self.repo:
                # Find the commit before the change
                commits = list(self.repo.iter_commits())
                if len(commits) > 1:
                    # Reset to previous commit
                    self.repo.head.reset(commits[1], working_tree=True)
                    logger.info(f"Rolled back to previous commit for change {change_id}")
            
            # Update change request status
            change_request.status = 'rolled_back'
            
            # Save updated change request
            self._save_change_request(change_request)
            
            # Log rollback
            self.audit_logger.log_action(
                user_id=rollback_user_id,
                action="change_rolled_back",
                table_name="change_control",
                record_id=change_id,
                change_reason=f"Rolled back by {rollback_user_id}: {rollback_reason}"
            )
            
            logger.info(f"Change {change_id} rolled back by {rollback_user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back change {change_id}: {e}")
            return False
    
    def get_change_history(self, component: Optional[str] = None, 
                          status: Optional[str] = None) -> List[ChangeRequest]:
        """
        Get change history
        
        Args:
            component: Filter by affected component
            status: Filter by change status
            
        Returns:
            List of change requests
        """
        try:
            changes_dir = Path("data/change_control")
            if not changes_dir.exists():
                return []
            
            change_requests = []
            for change_file in changes_dir.glob("*.json"):
                try:
                    with open(change_file, 'r') as f:
                        change_data = json.load(f)
                    
                    # Convert datetime strings back to datetime objects
                    change_data['created_at'] = datetime.fromisoformat(change_data['created_at'])
                    if change_data.get('implementation_date'):
                        change_data['implementation_date'] = datetime.fromisoformat(change_data['implementation_date'])
                    
                    change_request = ChangeRequest(**change_data)
                    
                    # Apply filters
                    if component and component not in change_request.affected_components:
                        continue
                    if status and change_request.status != status:
                        continue
                    
                    change_requests.append(change_request)
                    
                except Exception as e:
                    logger.warning(f"Error loading change request from {change_file}: {e}")
                    continue
            
            # Sort by creation date (newest first)
            change_requests.sort(key=lambda x: x.created_at, reverse=True)
            
            return change_requests
            
        except Exception as e:
            logger.error(f"Error getting change history: {e}")
            return []
    
    def _generate_change_id(self, user_id: str, change_type: str, description: str) -> str:
        """Generate unique change ID"""
        timestamp = datetime.now().isoformat()
        content = f"{user_id}_{change_type}_{description}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _assess_change_impact(self, change_type: str, affected_components: List[str], 
                            new_values: Dict[str, Any]) -> str:
        """Assess the impact of a change"""
        try:
            impact_score = 0
            
            # Impact based on change type
            type_impact = {
                'config': 1,
                'code': 3,
                'data': 2,
                'process': 2
            }
            impact_score += type_impact.get(change_type, 1)
            
            # Impact based on number of affected components
            impact_score += len(affected_components)
            
            # Impact based on complexity of changes
            impact_score += len(new_values) * 0.5
            
            if impact_score <= 2:
                return "Low"
            elif impact_score <= 4:
                return "Medium"
            elif impact_score <= 6:
                return "High"
            else:
                return "Critical"
                
        except Exception as e:
            logger.error(f"Error assessing change impact: {e}")
            return "Medium"
    
    def _generate_rollback_plan(self, old_values: Dict[str, Any], 
                               new_values: Dict[str, Any]) -> str:
        """Generate rollback plan"""
        try:
            rollback_steps = []
            
            for key, new_value in new_values.items():
                old_value = old_values.get(key, "Not set")
                rollback_steps.append(f"Restore {key} from {new_value} to {old_value}")
            
            return "; ".join(rollback_steps)
            
        except Exception as e:
            logger.error(f"Error generating rollback plan: {e}")
            return "Manual rollback required"
    
    def _can_approve_change(self, approver_id: str, change_request: ChangeRequest) -> bool:
        """Check if user can approve the change"""
        try:
            required_approvers = self.change_config['approval_workflow'].get(change_request.priority, [])
            
            # Check if approver is in the required list
            if approver_id in required_approvers:
                return True
            
            # Check if approver has already approved
            existing_approvals = [a['approver_id'] for a in change_request.approvals]
            if approver_id in existing_approvals:
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking approval permissions: {e}")
            return False
    
    def _get_required_approvals(self, priority: str) -> int:
        """Get number of required approvals for priority level"""
        try:
            required_approvers = self.change_config['approval_workflow'].get(priority, [])
            return min(len(required_approvers), self.change_config['approval_threshold'])
        except Exception as e:
            logger.error(f"Error getting required approvals: {e}")
            return 1
    
    def _save_change_request(self, change_request: ChangeRequest):
        """Save change request to file"""
        try:
            changes_dir = Path("data/change_control")
            changes_dir.mkdir(parents=True, exist_ok=True)
            
            change_file = changes_dir / f"{change_request.change_id}.json"
            
            # Convert to dictionary
            change_data = asdict(change_request)
            
            # Convert datetime objects to strings
            change_data['created_at'] = change_request.created_at.isoformat()
            if change_request.implementation_date:
                change_data['implementation_date'] = change_request.implementation_date.isoformat()
            
            with open(change_file, 'w') as f:
                json.dump(change_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving change request: {e}")
            raise
    
    def _load_change_request(self, change_id: str) -> Optional[ChangeRequest]:
        """Load change request from file"""
        try:
            change_file = Path("data/change_control") / f"{change_id}.json"
            
            if not change_file.exists():
                return None
            
            with open(change_file, 'r') as f:
                change_data = json.load(f)
            
            # Convert datetime strings back to datetime objects
            change_data['created_at'] = datetime.fromisoformat(change_data['created_at'])
            if change_data.get('implementation_date'):
                change_data['implementation_date'] = datetime.fromisoformat(change_data['implementation_date'])
            
            return ChangeRequest(**change_data)
            
        except Exception as e:
            logger.error(f"Error loading change request {change_id}: {e}")
            return None
    
    def _apply_changes_to_files(self, change_request: ChangeRequest):
        """Apply changes to actual files"""
        try:
            # This is a simplified implementation
            # In practice, this would apply changes to configuration files, code files, etc.
            
            if change_request.change_type == 'config':
                # Apply configuration changes
                self._apply_config_changes(change_request.new_values)
            elif change_request.change_type == 'code':
                # Apply code changes
                self._apply_code_changes(change_request.new_values)
            
            logger.info(f"Applied changes for request {change_request.change_id}")
            
        except Exception as e:
            logger.error(f"Error applying changes: {e}")
            raise
    
    def _apply_config_changes(self, new_values: Dict[str, Any]):
        """Apply configuration changes"""
        # Simplified implementation
        logger.info(f"Applying config changes: {new_values}")
    
    def _apply_code_changes(self, new_values: Dict[str, Any]):
        """Apply code changes"""
        # Simplified implementation
        logger.info(f"Applying code changes: {new_values}")
    
    def get_change_control_statistics(self) -> Dict[str, Any]:
        """Get change control statistics"""
        return {
            'total_changes': self.stats['total_changes'],
            'approved_changes': self.stats['approved_changes'],
            'rejected_changes': self.stats['rejected_changes'],
            'pending_changes': self.stats['pending_changes'],
            'change_control_efficiency_percentage': self.stats['change_control_efficiency'],
            'approval_rate_percentage': (self.stats['approved_changes'] / max(1, self.stats['total_changes'])) * 100
        } 

    def log_change(self, user_id: str, change_type: str, description: str, affected_components: list) -> dict:
        """
        Simulate logging a change for change control demo/testing.
        Returns a dictionary with change_id, status, and details.
        """
        import uuid
        change_id = str(uuid.uuid4())
        # In a real system, this would write to a database or file
        result = {
            'change_id': change_id,
            'user_id': user_id,
            'change_type': change_type,
            'description': description,
            'affected_components': affected_components,
            'status': 'logged',
            'timestamp': str(datetime.now())
        }
        # Optionally print or log the change
        print(f"[ChangeControl] Change logged: {result}")
        return result 