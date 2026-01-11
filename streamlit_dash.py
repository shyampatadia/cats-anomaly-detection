"""
CATS Anomaly Detection Dashboard
Real-time monitoring interface for multivariate time series anomaly detection.

Usage: streamlit run streamlit_dash.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# Page config
st.set_page_config(
    page_title="CATS Anomaly Detection",
    page_icon="⚠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3142;
        border-radius: 8px;
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .alert-box {
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid #ef4444;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_dashboard_data():
    data_path = "outputs/dashboard_data.json"
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            return json.load(f)
    else:
        st.error(f"Dashboard data not found at {data_path}. Run cats_pipeline.py first.")
        st.stop()
        return None

data = load_dashboard_data()

# Sidebar
st.sidebar.title("CATS Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Live Monitor", "EDA Analysis", "Model Comparison"],
    index=0
)

st.sidebar.markdown("---")

# Sidebar controls for Live Monitor
if page == "Live Monitor":
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["ensemble_proba", "rf_proba", "gb_proba", "iso_proba"],
        format_func=lambda x: {
            "ensemble_proba": "Ensemble",
            "rf_proba": "Random Forest",
            "gb_proba": "Gradient Boosting",
            "iso_proba": "Isolation Forest"
        }[x]
    )

    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

    num_points = st.sidebar.slider(
        "Time Window",
        min_value=50,
        max_value=200,
        value=100,
        step=10
    )

st.sidebar.markdown("---")
st.sidebar.info("**Dataset**: CATS\n\n**Samples**: 5M timestamps\n\n**Channels**: 17")

# ============================================================================
# LIVE MONITOR PAGE
# ============================================================================

if page == "Live Monitor":
    st.title("CATS Anomaly Detection System")
    st.markdown("Real-time multivariate time series monitoring")

    # Get data for visualization
    ts_data = data['time_series']

    # Create dataframe
    df_viz = pd.DataFrame(ts_data)

    # Calculate statistics
    scores = df_viz[model_choice].values
    true_labels = df_viz['true_label'].values

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Samples",
            value=f"{data['eda']['n_samples']:,}"
        )

    with col2:
        st.metric(
            label="Anomaly Rate",
            value=f"{data['eda']['anomaly_rate']:.1f}%"
        )

    with col3:
        detected = len([s for s in scores if s > threshold])
        st.metric(
            label="Detected Anomalies",
            value=detected
        )

    with col4:
        avg_score = np.mean(scores)
        st.metric(
            label="Average Score",
            value=f"{avg_score:.3f}"
        )

    # Main chart
    st.markdown("### Anomaly Score Over Time")

    # Select window
    start_idx = st.slider(
        "Start Position",
        min_value=0,
        max_value=max(0, len(df_viz) - num_points),
        value=0,
        step=10
    )

    df_window = df_viz.iloc[start_idx:start_idx + num_points]

    fig = go.Figure()

    # Score line
    fig.add_trace(go.Scatter(
        x=df_window['index'],
        y=df_window[model_choice],
        mode='lines',
        name='Anomaly Score',
        line=dict(color='#10b981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Threshold: {threshold}",
        annotation_position="right"
    )

    # Mark true anomalies
    anomaly_points = df_window[df_window['true_label'] == 1]
    if len(anomaly_points) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_points['index'],
            y=anomaly_points[model_choice],
            mode='markers',
            name='True Anomaly',
            marker=dict(color='#ef4444', size=8, symbol='x')
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(range=[0, 1], title="Anomaly Score"),
        xaxis=dict(title="Time Index"),
        showlegend=True,
        legend=dict(orientation='h', y=1.1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Two columns: Statistics and Alerts
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Detection Statistics")

        # Calculate detection metrics
        predicted = (scores > threshold).astype(int)
        true_positives = np.sum((predicted == 1) & (true_labels == 1))
        false_positives = np.sum((predicted == 1) & (true_labels == 0))
        false_negatives = np.sum((predicted == 0) & (true_labels == 1))
        true_negatives = np.sum((predicted == 0) & (true_labels == 0))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        metrics_df = pd.DataFrame({
            'Metric': ['True Positives', 'False Positives', 'False Negatives', 'True Negatives', 'Precision', 'Recall'],
            'Value': [
                true_positives,
                false_positives,
                false_negatives,
                true_negatives,
                f"{precision:.2%}",
                f"{recall:.2%}"
            ]
        })

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Alerts")

        # Find points above threshold
        alerts_df = df_viz[df_viz[model_choice] > threshold].head(10)

        if len(alerts_df) == 0:
            st.info("No alerts detected")
        else:
            for _, alert in alerts_df.iterrows():
                st.markdown(f"""
                <div class="alert-box">
                    <strong>Index {alert['index']}</strong><br/>
                    Score: <span style="color:#ef4444">{alert[model_choice]:.3f}</span>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# EDA PAGE
# ============================================================================

elif page == "EDA Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("Dataset statistics and feature analysis")

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", f"{data['eda']['n_samples']:,}")
    with col2:
        st.metric("Features", data['eda']['n_features'])
    with col3:
        st.metric("Anomalies", f"{data['eda']['n_anomalies']:,}")
    with col4:
        st.metric("Anomaly Rate", f"{data['eda']['anomaly_rate']:.2f}%")

    st.markdown("---")

    # Two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top Feature Correlations")

        corr_data = data['eda']['top_correlations']
        corr_df = pd.DataFrame(corr_data)

        fig = px.bar(
            corr_df,
            x='correlation',
            y='pair',
            orientation='h',
            color='correlation',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False,
            yaxis_title="",
            xaxis_title="Correlation"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Feature Importance")

        imp_data = data['feature_importance']
        imp_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in sorted(imp_data.items(), key=lambda x: x[1], reverse=True)[:8]
        ])

        fig = px.bar(
            imp_df,
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Teal'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False,
            yaxis_title="",
            xaxis_title="Importance"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Root cause and categories
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Root Cause Distribution")
        rc_data = data['eda']['root_cause_distribution']
        if rc_data:
            rc_df = pd.DataFrame([
                {'channel': k, 'count': v}
                for k, v in rc_data.items()
            ])

            fig = px.bar(
                rc_df,
                x='count',
                y='channel',
                orientation='h',
                color='count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No root cause data available")

    with col2:
        st.markdown("### Channel Categories")

        cat_data = {
            'Commands (4)': 'aimp, amud, adbr, adfl',
            'Environmental (3)': 'arnd, asin1, asin2',
            'Telemetry (10)': 'bed1, bed2, bfo1, bfo2, bso1, bso2, bso3, ced1, cfo1, cso1'
        }

        for cat, channels in cat_data.items():
            with st.expander(cat, expanded=False):
                st.write(channels)

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    st.markdown("Evaluation metrics for all trained models")

    models_data = data['models']

    # Find best model
    best_model = max(models_data.items(), key=lambda x: x[1]['roc_auc'])[0]

    # Model cards
    cols = st.columns(3)

    for i, (name, metrics) in enumerate(models_data.items()):
        with cols[i]:
            is_best = name == best_model

            display_name = name.replace('_', ' ').title()
            if is_best:
                st.success(f"**{display_name}** (Best)")
            else:
                st.info(f"**{display_name}**")

            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            st.metric("F1 Score", f"{metrics['f1']:.1f}%")
            st.metric("Precision", f"{metrics['precision']:.1f}%")
            st.metric("Recall", f"{metrics['recall']:.1f}%")

    st.markdown("---")

    # Comparison chart
    st.markdown("### Performance Metrics")

    metrics_df = pd.DataFrame([
        {
            'Model': name.replace('_', ' ').title(),
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1': m['f1'],
            'AUC': m['roc_auc'] * 100
        }
        for name, m in models_data.items()
    ])

    fig = go.Figure()

    colors = {'Precision': '#3b82f6', 'Recall': '#10b981', 'F1': '#f59e0b'}

    for metric in ['Precision', 'Recall', 'F1']:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric],
            marker_color=colors[metric]
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='group',
        height=400,
        yaxis_title='Score (%)',
        legend=dict(orientation='h', y=1.1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Key findings
    st.markdown("### Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**Best Model**: {best_model.replace('_', ' ').title()} with AUC {models_data[best_model]['roc_auc']:.4f}")
        st.info("**Top Features**: bfo2, cso1, bso3 contribute most to detection")

    with col2:
        st.info("**Class Imbalance**: Handled with balanced class weights")
        st.info("**Ensemble**: Combines all models for robust predictions")

# Footer
st.markdown("---")
st.markdown("CATS Anomaly Detection System | Built with Streamlit")
