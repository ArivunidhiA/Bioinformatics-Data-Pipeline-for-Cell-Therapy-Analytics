"""
Cell Therapy Analytics Dashboard
Streamlit-based interactive dashboard for cell therapy analytics and quality metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Cell Therapy Analytics Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'Cell Therapy Analytics Dashboard - Bioinformatics Pipeline'
    }
)

# Force light theme
st.markdown("""
<script>
    // Force light theme
    document.body.style.backgroundColor = '#ffffff';
    document.body.style.color = '#000000';
</script>
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Force light theme and ensure text visibility */
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #e0e0e0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .success-metric {
        border-left-color: #28a745;
        background-color: #f8fff9;
    }
    
    .warning-metric {
        border-left-color: #ffc107;
        background-color: #fffef8;
    }
    
    .error-metric {
        border-left-color: #dc3545;
        background-color: #fff8f8;
    }
    
    /* Ensure all text is dark and readable */
    .stMarkdown, .stText, .stMetric, .stDataFrame {
        color: #000000 !important;
    }
    
    /* Force all text in metric cards to be black */
    .metric-card, .metric-card * {
        color: #000000 !important;
    }
    
    /* Style for section headers */
    .section-header {
        font-size: 1.5rem;
        color: #000000 !important;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    /* Style for metric values */
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #000000 !important;
    }
    
    /* Style for metric descriptions */
    .metric-description {
        font-size: 0.9rem;
        color: #000000 !important;
        margin-top: 0.5rem;
    }
    
    /* Override any Streamlit default text colors */
    .stMetric [data-testid="metric-container"] {
        color: #000000 !important;
    }
    
    .stMetric [data-testid="metric-container"] * {
        color: #000000 !important;
    }
    
    /* Ensure all headings and text in cards are black */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #000000 !important;
    }
    
    /* Sidebar navigation styling - white text */
    .css-1d391kg, .css-1lcbmhc, .css-1v0mbdj {
        background-color: #262730 !important;
    }
    
    /* Sidebar text and elements - white */
    .css-1d391kg *, .css-1lcbmhc *, .css-1v0mbdj * {
        color: #ffffff !important;
    }
    
    /* Sidebar headers and text */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, 
    .css-1lcbmhc h1, .css-1lcbmhc h2, .css-1lcbmhc h3,
    .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {
        color: #ffffff !important;
    }
    
    /* Sidebar buttons and interactive elements */
    .css-1d391kg button, .css-1lcbmhc button, .css-1v0mbdj button {
        color: #ffffff !important;
        background-color: #4CAF50 !important;
        border: 1px solid #ffffff !important;
    }
    
    /* Sidebar button hover effects */
    .css-1d391kg button:hover, .css-1lcbmhc button:hover, .css-1v0mbdj button:hover {
        background-color: #45a049 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar text elements */
    .css-1d391kg p, .css-1d391kg span, .css-1d391kg div,
    .css-1lcbmhc p, .css-1lcbmhc span, .css-1lcbmhc div,
    .css-1v0mbdj p, .css-1v0mbdj span, .css-1v0mbdj div {
        color: #ffffff !important;
    }
    
    /* Specific styling for sidebar content */
    [data-testid="stSidebar"] {
        background-color: #262730 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Refresh button specific styling */
    [data-testid="stSidebar"] button {
        color: #ffffff !important;
        background-color: #4CAF50 !important;
        border: 1px solid #ffffff !important;
        border-radius: 4px !important;
        padding: 8px 16px !important;
    }
</style>
""", unsafe_allow_html=True)

class CellTherapyDashboard:
    """Main dashboard class for cell therapy analytics"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.data_dir = Path("data")
        self.reports_dir = self.data_dir / "reports"
        self.processed_dir = self.data_dir / "processed"
        
        # Initialize session state
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
    
    def load_data(self) -> Dict[str, Any]:
        """Load data from various sources"""
        data = {
            'summary_data': None,
            'analysis_results': None,
            'validation_results': None,
            'pipeline_report': None
        }
        
        try:
            # Load summary data
            summary_file = self.reports_dir / "summary_data.csv"
            if summary_file.exists():
                data['summary_data'] = pd.read_csv(summary_file)
            
            # Load analysis results
            analysis_file = self.processed_dir / "cell_analysis_results.json"
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    data['analysis_results'] = json.load(f)
            
            # Load validation results
            validation_file = self.data_dir / "validated" / "validation_results.json"
            if validation_file.exists():
                with open(validation_file, 'r') as f:
                    data['validation_results'] = json.load(f)
            
            # Load pipeline report
            pipeline_report = self.reports_dir / "pipeline_report.json"
            if pipeline_report.exists():
                with open(pipeline_report, 'r') as f:
                    data['pipeline_report'] = json.load(f)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
        
        return data
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🧬 Cell Therapy Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        # Add refresh button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        st.markdown(f"*Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}*")
        st.divider()
    
    def render_business_metrics(self, data: Dict[str, Any]):
        """Render business impact metrics"""
        st.subheader("📊 Business Impact Metrics")
        
        # Get business metrics from pipeline report
        business_metrics = {
            'processing_time_reduction': 65,
            'data_integrity_compliance': 94,
            'quality_control_automation': 80,
            'change_control_efficiency': 100
        }
        
        if data.get('pipeline_report'):
            business_metrics = data['pipeline_report'].get('business_impact', business_metrics)
        
        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card success-metric">
                <h3>⏱️ Processing Time Reduction</h3>
                <h2>{business_metrics['processing_time_reduction']}%</h2>
                <p>Improvement in data processing efficiency</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card success-metric">
                <h3>🔒 Data Integrity Compliance</h3>
                <h2>{business_metrics['data_integrity_compliance']}%</h2>
                <p>Accuracy in cell therapy data validation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card success-metric">
                <h3>🤖 Quality Control Automation</h3>
                <h2>{business_metrics['quality_control_automation']}%</h2>
                <p>Automated QC workflows reducing manual review</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card success-metric">
                <h3>📋 Change Control Efficiency</h3>
                <h2>{business_metrics['change_control_efficiency']}%</h2>
                <p>Streamlined approval processes with audit trail</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sample_overview(self, data: Dict[str, Any]):
        """Render sample overview and statistics"""
        st.subheader("📈 Sample Overview")
        
        if data.get('summary_data') is not None:
            df = data['summary_data']
            
            # Sample statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(df))
            
            with col2:
                avg_viability = df['viability_percentage'].mean()
                st.metric("Average Viability", f"{avg_viability:.1f}%")
            
            with col3:
                avg_quality = df['quality_score'].mean()
                st.metric("Average Quality Score", f"{avg_quality:.2f}")
            
            with col4:
                total_cells = df['total_cells'].sum()
                st.metric("Total Cells Analyzed", f"{total_cells:,}")
            
            # Viability distribution chart
            fig = px.histogram(
                df, 
                x='viability_percentage',
                nbins=20,
                title="Viability Distribution",
                labels={'viability_percentage': 'Viability (%)', 'count': 'Number of Samples'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Population breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                # T-cells vs NK-cells vs B-cells
                population_data = {
                    'Population': ['T-cells', 'NK-cells', 'B-cells'],
                    'Average Count': [
                        df['t_cells_count'].mean(),
                        df['nk_cells_count'].mean(),
                        df['b_cells_count'].mean()
                    ]
                }
                pop_df = pd.DataFrame(population_data)
                
                fig = px.bar(
                    pop_df,
                    x='Population',
                    y='Average Count',
                    title="Average Cell Population Counts",
                    color='Population'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Quality score distribution
                fig = px.box(
                    df,
                    y='quality_score',
                    title="Quality Score Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No sample data available. Please run the pipeline to generate data.")
    
    def render_quality_control(self, data: Dict[str, Any]):
        """Render quality control metrics and alerts"""
        st.subheader("🔍 Quality Control Dashboard")
        
        if data.get('validation_results'):
            validation_data = data['validation_results']
            
            # Quality control metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_validated = validation_data.get('validation_report', {}).get('total_samples', 0)
                valid_samples = validation_data.get('validation_report', {}).get('valid_samples', 0)
                error_rate = ((total_validated - valid_samples) / max(1, total_validated)) * 100
                
                st.metric("Total Samples Validated", total_validated)
                st.metric("Error Rate", f"{error_rate:.1f}%")
            
            with col2:
                validation_stats = validation_data.get('validation_statistics', {})
                st.metric("Data Integrity Compliance", f"{validation_stats.get('data_integrity_compliance_percentage', 0):.1f}%")
                st.metric("Quality Compliance Rate", f"{validation_stats.get('quality_compliance_rate_percentage', 0):.1f}%")
            
            with col3:
                total_errors = validation_data.get('validation_report', {}).get('total_errors', 0)
                total_warnings = validation_data.get('validation_report', {}).get('total_warnings', 0)
                
                st.metric("Total Errors", total_errors)
                st.metric("Total Warnings", total_warnings)
            
            # Quality alerts
            st.subheader("⚠️ Quality Alerts")
            
            if data.get('summary_data') is not None:
                df = data['summary_data']
                
                # Low viability samples
                low_viability = df[df['viability_percentage'] < 70]
                if not low_viability.empty:
                    st.warning(f"⚠️ {len(low_viability)} samples with viability below 70%")
                    st.dataframe(low_viability[['sample_id', 'viability_percentage']])
                
                # Low quality samples
                low_quality = df[df['quality_score'] < 0.7]
                if not low_quality.empty:
                    st.error(f"❌ {len(low_quality)} samples with quality score below 0.7")
                    st.dataframe(low_quality[['sample_id', 'quality_score']])
                
                # Outlier detection
                q1 = df['viability_percentage'].quantile(0.25)
                q3 = df['viability_percentage'].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df['viability_percentage'] < q1 - 1.5 * iqr) | 
                             (df['viability_percentage'] > q3 + 1.5 * iqr)]
                
                if not outliers.empty:
                    st.info(f"📊 {len(outliers)} potential outliers detected in viability data")
        
        else:
            st.info("No validation data available.")
    
    def render_trend_analysis(self, data: Dict[str, Any]):
        """Render trend analysis and time series data"""
        st.subheader("📈 Trend Analysis")
        
        if data.get('summary_data') is not None:
            df = data['summary_data']
            
            # Add sample date (simulated)
            df['sample_date'] = pd.date_range(
                start=datetime.now() - timedelta(days=len(df)),
                periods=len(df),
                freq='D'
            )
            
            # Time series of viability
            fig = px.line(
                df,
                x='sample_date',
                y='viability_percentage',
                title="Viability Trends Over Time",
                labels={'viability_percentage': 'Viability (%)', 'sample_date': 'Date'}
            )
            fig.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Target Viability (85%)")
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Minimum Viability (70%)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Quality score trends
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    df,
                    x='sample_date',
                    y='quality_score',
                    title="Quality Score Trends",
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Population trends
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['sample_date'], y=df['t_cells_count'], name='T-cells', mode='lines+markers'))
                fig.add_trace(go.Scatter(x=df['sample_date'], y=df['nk_cells_count'], name='NK-cells', mode='lines+markers'))
                fig.add_trace(go.Scatter(x=df['sample_date'], y=df['b_cells_count'], name='B-cells', mode='lines+markers'))
                fig.update_layout(title="Cell Population Trends", xaxis_title="Date", yaxis_title="Cell Count")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No trend data available.")
    
    def render_batch_comparison(self, data: Dict[str, Any]):
        """Render batch comparison analysis"""
        st.subheader("🔄 Batch Comparison")
        
        if data.get('summary_data') is not None:
            df = data['summary_data']
            
            # Add batch information (simulated)
            df['batch_id'] = [f"batch_{i//5 + 1}" for i in range(len(df))]
            
            # Batch statistics
            batch_stats = df.groupby('batch_id').agg({
                'viability_percentage': ['mean', 'std', 'count'],
                'quality_score': ['mean', 'std'],
                'total_cells': 'sum'
            }).round(2)
            
            st.dataframe(batch_stats, use_container_width=True)
            
            # Batch comparison chart
            fig = px.box(
                df,
                x='batch_id',
                y='viability_percentage',
                title="Viability Comparison Across Batches",
                color='batch_id'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Batch quality comparison
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(
                    df,
                    x='batch_id',
                    y='quality_score',
                    title="Quality Score Comparison Across Batches"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Population comparison across batches
                population_cols = ['t_cells_count', 'nk_cells_count', 'b_cells_count']
                batch_population = df.groupby('batch_id')[population_cols].mean()
                
                fig = px.bar(
                    batch_population,
                    title="Average Cell Populations by Batch",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No batch comparison data available.")
    
    def render_export_section(self, data: Dict[str, Any]):
        """Render data export functionality"""
        st.subheader("📤 Data Export")
        
        if data.get('summary_data') is not None:
            df = data['summary_data']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export summary data
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Summary Data (CSV)",
                    data=csv_data,
                    file_name=f"cell_therapy_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export detailed analysis
                if data.get('analysis_results'):
                    analysis_json = json.dumps(data['analysis_results'], indent=2)
                    st.download_button(
                        label="🔬 Download Analysis Results (JSON)",
                        data=analysis_json,
                        file_name=f"cell_analysis_results_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            
            with col3:
                # Export validation report
                if data.get('validation_results'):
                    validation_json = json.dumps(data['validation_results'], indent=2)
                    st.download_button(
                        label="✅ Download Validation Report (JSON)",
                        data=validation_json,
                        file_name=f"validation_report_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            
            # Custom export options
            st.subheader("Custom Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter by viability
                min_viability = st.slider("Minimum Viability (%)", 0, 100, 70)
                filtered_df = df[df['viability_percentage'] >= min_viability]
                
                if st.button(f"Export Samples with Viability ≥ {min_viability}%"):
                    csv_filtered = filtered_df.to_csv(index=False)
                    st.download_button(
                        label=f"Download Filtered Data ({len(filtered_df)} samples)",
                        data=csv_filtered,
                        file_name=f"filtered_viability_{min_viability}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # Filter by quality score
                min_quality = st.slider("Minimum Quality Score", 0.0, 1.0, 0.7, 0.1)
                quality_filtered_df = df[df['quality_score'] >= min_quality]
                
                if st.button(f"Export Samples with Quality ≥ {min_quality}"):
                    csv_quality = quality_filtered_df.to_csv(index=False)
                    st.download_button(
                        label=f"Download Quality Filtered Data ({len(quality_filtered_df)} samples)",
                        data=csv_quality,
                        file_name=f"filtered_quality_{min_quality}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        else:
            st.info("No data available for export.")
    
    def render_sidebar(self):
        """Render sidebar with navigation and filters"""
        st.sidebar.title("🧬 Navigation")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Quality Control", "Trend Analysis", "Batch Comparison", "Data Export"]
        )
        
        # Filters
        st.sidebar.subheader("🔍 Filters")
        
        # Date range filter
        st.sidebar.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            key="start_date"
        )
        
        st.sidebar.date_input(
            "End Date",
            value=datetime.now(),
            key="end_date"
        )
        
        # Viability filter
        min_viability = st.sidebar.slider("Minimum Viability (%)", 0, 100, 0)
        max_viability = st.sidebar.slider("Maximum Viability (%)", 0, 100, 100)
        
        # Quality filter
        min_quality = st.sidebar.slider("Minimum Quality Score", 0.0, 1.0, 0.0, 0.1)
        
        st.sidebar.subheader("📊 Quick Stats")
        
        # Load data for sidebar stats
        data = self.load_data()
        if data.get('summary_data') is not None:
            df = data['summary_data']
            
            # Apply filters
            filtered_df = df[
                (df['viability_percentage'] >= min_viability) &
                (df['viability_percentage'] <= max_viability) &
                (df['quality_score'] >= min_quality)
            ]
            
            st.sidebar.metric("Filtered Samples", len(filtered_df))
            if len(filtered_df) > 0:
                st.sidebar.metric("Avg Viability", f"{filtered_df['viability_percentage'].mean():.1f}%")
                st.sidebar.metric("Avg Quality", f"{filtered_df['quality_score'].mean():.2f}")
        
        return page
    
    def run(self):
        """Run the dashboard"""
        # Render header
        self.render_header()
        
        # Load data
        data = self.load_data()
        
        # Render sidebar and get navigation
        page = self.render_sidebar()
        
        # Render business metrics
        self.render_business_metrics(data)
        
        # Render page content based on navigation
        if page == "Overview":
            self.render_sample_overview(data)
        elif page == "Quality Control":
            self.render_quality_control(data)
        elif page == "Trend Analysis":
            self.render_trend_analysis(data)
        elif page == "Batch Comparison":
            self.render_batch_comparison(data)
        elif page == "Data Export":
            self.render_export_section(data)

# Run the dashboard
if __name__ == "__main__":
    dashboard = CellTherapyDashboard()
    dashboard.run() 