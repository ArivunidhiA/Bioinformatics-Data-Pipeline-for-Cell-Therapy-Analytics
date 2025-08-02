# Standard Operating Procedure (SOP) 001
## Data Ingestion for Cell Therapy Analytics Pipeline

**Document Version:** 1.0  
**Effective Date:** January 1, 2024  
**Review Date:** January 1, 2025  
**Author:** Bioinformatics Team  
**Approved By:** Quality Assurance Manager  

---

## 1. Purpose and Scope

### 1.1 Purpose
This SOP establishes standardized procedures for data ingestion into the Cell Therapy Analytics Pipeline, ensuring data integrity, quality control, and regulatory compliance.

### 1.2 Scope
This procedure applies to all personnel involved in:
- Flow cytometry data ingestion
- Cell count data processing
- Sample metadata management
- Quality control validation
- Change control procedures

### 1.3 Regulatory Framework
- 21 CFR Part 11 - Electronic Records
- ALCOA+ Principles for Data Integrity
- GxP Guidelines for Life Sciences
- ISO 13485 Quality Management Systems

---

## 2. Definitions and Abbreviations

### 2.1 Definitions
- **FCS File**: Flow Cytometry Standard file format
- **Data Integrity**: Accuracy, completeness, and consistency of data
- **Validation**: Confirmation that data meets specified requirements
- **Audit Trail**: Chronological record of all data modifications
- **Change Control**: Systematic approach to managing changes to the system

### 2.2 Abbreviations
- **SOP**: Standard Operating Procedure
- **QC**: Quality Control
- **FCS**: Flow Cytometry Standard
- **GxP**: Good Practice Guidelines
- **ALCOA+**: Attributable, Legible, Contemporaneous, Original, Accurate, Complete, Consistent, Enduring, Available

---

## 3. Responsibilities

### 3.1 Data Ingestion Specialist
- Execute data ingestion procedures
- Perform quality control checks
- Document all activities
- Report issues and deviations

### 3.2 Quality Control Analyst
- Review validation results
- Approve or reject data based on quality criteria
- Maintain quality control documentation
- Escalate quality issues

### 3.3 System Administrator
- Maintain system infrastructure
- Monitor system performance
- Implement change control procedures
- Ensure data security

### 3.4 Quality Assurance Manager
- Review and approve SOPs
- Conduct periodic audits
- Ensure regulatory compliance
- Approve change requests

---

## 4. Equipment and Materials

### 4.1 Required Equipment
- Computer workstation with approved software
- Network access to data storage systems
- Backup storage devices
- Quality control monitoring tools

### 4.2 Required Software
- Cell Therapy Analytics Pipeline
- Flow cytometry analysis software
- Data validation tools
- Audit logging system

### 4.3 Required Documentation
- Data ingestion logbook
- Quality control checklists
- Change control forms
- Deviation reports

---

## 5. Safety Considerations

### 5.1 Data Security
- Ensure secure data transmission
- Maintain data confidentiality
- Follow access control procedures
- Protect against data loss

### 5.2 System Security
- Use approved authentication methods
- Maintain secure network connections
- Follow cybersecurity protocols
- Report security incidents

---

## 6. Procedure

### 6.1 Pre-Ingestion Preparation

#### 6.1.1 System Check
1. Verify system availability and performance
2. Check network connectivity
3. Validate user access permissions
4. Confirm backup systems are operational

#### 6.1.2 Data Source Verification
1. Verify data source authenticity
2. Check file format compliance
3. Validate file integrity
4. Confirm metadata completeness

#### 6.1.3 Quality Control Setup
1. Load validation rules
2. Configure quality thresholds
3. Set up monitoring alerts
4. Prepare audit logging

### 6.2 Data Ingestion Process

#### 6.2.1 Flow Cytometry Data Ingestion

**Step 1: File Validation**
```
1.1. Check file format (.fcs extension)
1.2. Verify file size and integrity
1.3. Validate file header information
1.4. Check for required parameters (FSC-A, SSC-A)
1.5. Document validation results
```

**Step 2: Data Extraction**
```
2.1. Parse FCS file using approved software
2.2. Extract measurement data
2.3. Validate data ranges and formats
2.4. Check for missing or corrupted data
2.5. Generate extraction report
```

**Step 3: Quality Control**
```
3.1. Apply automated quality checks
3.2. Validate event count (10,000 - 1,000,000)
3.3. Check parameter ranges
3.4. Identify outliers and anomalies
3.5. Document quality control results
```

**Step 4: Data Processing**
```
4.1. Apply gating strategies
4.2. Calculate viability metrics
4.3. Identify cell populations
4.4. Generate statistical summaries
4.5. Create processed data files
```

#### 6.2.2 Cell Count Data Ingestion

**Step 1: Data Format Validation**
```
1.1. Verify CSV/Excel format compliance
1.2. Check required column headers
1.3. Validate data types and ranges
1.4. Confirm calculation formulas
1.5. Document format validation
```

**Step 2: Content Validation**
```
2.1. Check for missing values
2.2. Validate numerical ranges
3.3. Verify calculation accuracy
2.4. Cross-reference with metadata
2.5. Document content validation
```

#### 6.2.3 Metadata Ingestion

**Step 1: Metadata Validation**
```
1.1. Verify required fields are present
1.2. Check data format compliance
1.3. Validate date formats
1.4. Confirm sample ID uniqueness
1.5. Document metadata validation
```

**Step 2: Cross-Reference Validation**
```
2.1. Match metadata with data files
2.2. Verify sample ID consistency
2.3. Check protocol version compatibility
2.4. Validate technician information
2.5. Document cross-reference results
```

### 6.3 Quality Control Procedures

#### 6.3.1 Automated Quality Checks
```
QC1. Minimum event count validation
QC2. Viability percentage range check (0-100%)
QC3. Data type validation
QC4. Format compliance verification
QC5. Statistical outlier detection
```

#### 6.3.2 Manual Quality Review
```
MQC1. Review automated QC results
MQC2. Investigate flagged issues
MQC3. Apply business rule validation
MQC4. Document manual review decisions
MQC5. Escalate unresolved issues
```

### 6.4 Data Storage and Archival

#### 6.4.1 Primary Storage
```
1. Store processed data in approved database
2. Create backup copies
3. Update audit trail
4. Generate storage confirmation
5. Document storage location
```

#### 6.4.2 Archival Procedures
```
1. Archive raw data files
2. Create archival index
3. Verify archival integrity
4. Update archival log
5. Document archival completion
```

---

## 7. Change Control Procedures

### 7.1 Change Request Process

#### 7.1.1 Change Request Initiation
```
1. Identify need for change
2. Document change justification
3. Assess change impact
4. Submit change request form
5. Assign change request number
```

#### 7.1.2 Change Assessment
```
1. Review change request
2. Evaluate technical feasibility
3. Assess regulatory impact
4. Estimate resource requirements
5. Determine approval level
```

#### 7.1.3 Change Approval
```
1. Route to appropriate approvers
2. Review change documentation
3. Approve or reject change
4. Document approval decision
5. Update change request status
```

#### 7.1.4 Change Implementation
```
1. Schedule change implementation
2. Prepare implementation plan
3. Execute change procedures
4. Validate change results
5. Update system documentation
```

### 7.2 Change Control Documentation

#### 7.2.1 Required Documentation
- Change request form
- Impact assessment report
- Approval documentation
- Implementation plan
- Validation results
- Post-implementation review

#### 7.2.2 Change Control Log
- Change request number
- Change description
- Requestor information
- Approval status
- Implementation date
- Validation results

---

## 8. Validation Protocols

### 8.1 Data Validation

#### 8.1.1 Input Validation
```
V1. File format validation
V2. Data type validation
V3. Range validation
V4. Completeness validation
V5. Consistency validation
```

#### 8.1.2 Processing Validation
```
V6. Algorithm validation
V7. Calculation accuracy
V8. Statistical validation
V9. Performance validation
V10. Output validation
```

### 8.2 System Validation

#### 8.2.1 Performance Validation
```
PV1. Processing speed validation
PV2. Throughput capacity validation
PV3. Memory usage validation
PV4. Network performance validation
PV5. Scalability validation
```

#### 8.2.2 Security Validation
```
SV1. Access control validation
SV2. Data encryption validation
SV3. Audit trail validation
SV4. Backup and recovery validation
SV5. Disaster recovery validation
```

---

## 9. Quality Assurance

### 9.1 Quality Metrics

#### 9.1.1 Data Quality Metrics
- Data integrity compliance: ≥94%
- Processing accuracy: ≥99%
- Validation success rate: ≥95%
- Error detection rate: ≥90%

#### 9.1.2 System Quality Metrics
- System uptime: ≥99.5%
- Processing time reduction: ≥65%
- Quality control automation: ≥80%
- Change control efficiency: 100%

### 9.2 Quality Monitoring

#### 9.2.1 Continuous Monitoring
```
1. Real-time quality metrics
2. Automated alert systems
3. Performance dashboards
4. Error tracking and reporting
5. Trend analysis and reporting
```

#### 9.2.2 Periodic Reviews
```
1. Monthly quality reviews
2. Quarterly performance assessments
3. Annual system audits
4. Regulatory compliance reviews
5. User satisfaction surveys
```

---

## 10. Documentation and Record Keeping

### 10.1 Required Documentation

#### 10.1.1 Process Documentation
- Data ingestion logs
- Quality control reports
- Validation results
- Change control records
- Audit trail logs

#### 10.1.2 System Documentation
- System configuration
- User access logs
- Performance metrics
- Error logs
- Maintenance records

### 10.2 Record Retention

#### 10.2.1 Retention Periods
- Process records: 7 years
- System logs: 7 years
- Audit trails: 7 years
- Quality reports: 7 years
- Change control records: 7 years

#### 10.2.2 Archival Procedures
```
1. Identify records for archival
2. Create archival index
3. Transfer to archival storage
4. Verify archival integrity
5. Update archival log
```

---

## 11. Training and Competency

### 11.1 Training Requirements

#### 11.1.1 Initial Training
```
1. System overview and architecture
2. Data ingestion procedures
3. Quality control protocols
4. Change control procedures
5. Regulatory compliance requirements
```

#### 11.1.2 Ongoing Training
```
1. System updates and changes
2. New procedures and protocols
3. Regulatory updates
4. Best practices and improvements
5. Incident response procedures
```

### 11.2 Competency Assessment

#### 11.2.1 Assessment Criteria
- Technical knowledge and skills
- Procedural compliance
- Quality control performance
- Problem-solving abilities
- Regulatory awareness

#### 11.2.2 Assessment Frequency
- Initial competency assessment
- Annual competency review
- Post-training assessment
- Incident-based assessment
- Performance-based assessment

---

## 12. Deviation Management

### 12.1 Deviation Categories

#### 12.1.1 Minor Deviations
- Procedural deviations with no impact on data quality
- System performance issues with workarounds
- Documentation delays within acceptable limits

#### 12.1.2 Major Deviations
- Data quality issues requiring investigation
- System failures affecting multiple users
- Regulatory compliance concerns
- Security incidents

### 12.2 Deviation Procedures

#### 12.2.1 Deviation Reporting
```
1. Identify and document deviation
2. Assess deviation impact
3. Implement immediate corrective actions
4. Report deviation to supervisor
5. Initiate investigation if required
```

#### 12.2.2 Deviation Investigation
```
1. Conduct root cause analysis
2. Identify contributing factors
3. Develop corrective actions
4. Implement preventive measures
5. Document investigation results
```

---

## 13. References

### 13.1 Regulatory References
- 21 CFR Part 11 - Electronic Records; Electronic Signatures
- FDA Guidance for Industry: Computerized Systems Used in Clinical Investigations
- ICH E6(R2) Good Clinical Practice
- ISO 13485:2016 Medical devices - Quality management systems

### 13.2 Technical References
- Flow Cytometry Standard (FCS) File Format
- Cell Therapy Analytics Pipeline Technical Specifications
- Database Schema Documentation
- API Documentation

### 13.3 Procedural References
- Quality Management System Manual
- Change Control Procedures
- Validation Master Plan
- Training Program Documentation

---

## 14. Appendices

### Appendix A: Data Ingestion Checklist
### Appendix B: Quality Control Forms
### Appendix C: Change Request Templates
### Appendix D: Validation Protocols
### Appendix E: Training Materials

---

**Document Control:**
- **Version History:** See document control log
- **Distribution:** Quality Assurance, Bioinformatics Team, System Administrators
- **Review Schedule:** Annual review or as needed based on changes
- **Next Review Date:** January 1, 2025 