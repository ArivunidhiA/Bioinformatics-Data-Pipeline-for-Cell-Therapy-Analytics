"""
Cell Therapy Analytics Pipeline
A comprehensive bioinformatics data pipeline for cell therapy analytics
"""

__version__ = "1.0.0"
__author__ = "Bioinformatics Team"
__description__ = "Cell Therapy Analytics Pipeline for Life Sciences"

from .data_ingestion import FlowCytometryReader, DataValidator
from .processing import CellAnalysis, QualityControl, StatisticalAnalysis
from .utils import ChangeControl, AuditLogging

__all__ = [
    'FlowCytometryReader',
    'DataValidator', 
    'CellAnalysis',
    'QualityControl',
    'StatisticalAnalysis',
    'ChangeControl',
    'AuditLogging'
] 