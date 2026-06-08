# ============================================================
# 🌿 ADVANCED ASD FIELD SPEC 4 HYPERSPECTRAL DASHBOARD
# ============================================================

import os
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.signal import savgol_filter
from specdal import Spectrum


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ASD Spectral Signature Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1B5E20;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 25px;
    }
    .info-box {
        background-color: #F1F8E9;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #43A047;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR MENU
# ============================================================

st.sidebar.title("🌿 Remote Sensing & GIS")
st.sidebar.caption("🌍 Hyperspectral ASD AI Research Platform")
st.sidebar.success("✅ System Ready")
st.sidebar.markdown("---")

main_menu = st.sidebar.radio(
    "📂 Navigation Menu",
    [
        "🏠 Dashboard",
        "📊 Spectral Analysis",
        "📂 Multi-ASD Comparison",
        "🧠 Dataset Builder",
        "📈 Derivative Spectroscopy",
        "🔥 Spectral Heatmap",
        "🌿 Vegetation Stress",
        "📚 Documentation",
        "👨‍💻 Profile Detail"
    ],
    key="main_navigation_menu"
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚙ Visualization Settings")

apply_smoothing = st.sidebar.checkbox(
    "Apply Savitzky-Golay Smoothing",
    value=True
)

show_original = st.sidebar.checkbox(
    "Show Original Curve",
    value=True
)

window_length = st.sidebar.slider(
    "Smoothing Window",
    min_value=5,
    max_value=51,
    value=21,
    step=2
)

poly_order = st.sidebar.slider(
    "Polynomial Order",
    min_value=1,
    max_value=5,
    value=3
)

st.sidebar.markdown("---")

st.sidebar.subheader("🌈 ASD Spectral Regions")

st.sidebar.info(
    """
    🔵 VIS : 400–700 nm  
    🌿 Red Edge : 680–750 nm  
    🌱 NIR : 700–1300 nm  
    💧 SWIR : 1300–2500 nm
    """
)


# ============================================================
# FUNCTIONS
# ============================================================

def save_temp_file(uploaded_file):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_path


def load_asd_file(asd_path):
    spectrum = Spectrum(filepath=asd_path)

    wavelength = spectrum.measurement.index.values.astype(float)
    reflectance = spectrum.measurement.values.astype(float)

    reflectance = np.ravel(reflectance)

    df = pd.DataFrame({
        "Wavelength": wavelength,
        "Reflectance": reflectance
    })

    df = df.dropna()

    return df["Wavelength"].values, df["Reflectance"].values, df


def apply_smoothing_filter(reflectance):
    if not apply_smoothing:
        return reflectance

    valid_window = window_length

    if valid_window >= len(reflectance):
        valid_window = len(reflectance) - 1

    if valid_window % 2 == 0:
        valid_window -= 1

    if valid_window <= poly_order:
        valid_window = poly_order + 2

    if valid_window % 2 == 0:
        valid_window += 1

    return savgol_filter(
        reflectance,
        valid_window,
        poly_order
    )


def calculate_indices(wavelength, reflectance):
    def get_band_value(target_nm):
        idx = np.argmin(np.abs(wavelength - target_nm))
        return reflectance[idx]

    blue = get_band_value(490)
    green = get_band_value(560)
    red = get_band_value(665)
    red_edge = get_band_value(705)
    nir = get_band_value(842)
    swir1 = get_band_value(1610)

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndwi = (green - nir) / (green + nir + 1e-10)
    ndre = (nir - red_edge) / (nir + red_edge + 1e-10)
    msi = swir1 / (nir + 1e-10)

    index_df = pd.DataFrame({
        "Index": ["NDVI", "NDWI", "NDRE", "MSI"],
        "Value": [ndvi, ndwi, ndre, msi]
    })

    return index_df


def create_spectral_plot(wavelength, reflectance, smooth_reflectance, title):
    fig = go.Figure()

    if show_original:
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=reflectance,
                mode="lines",
                name="Original Spectrum",
                line=dict(width=1)
            )
        )

    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=smooth_reflectance,
            mode="lines",
            name="Smoothed Spectrum",
            line=dict(width=3)
        )
    )

    spectral_bands = {
        "Blue": 490,
        "Green": 560,
        "Red": 665,
        "Red Edge": 705,
        "NIR": 842,
        "SWIR1": 1610,
        "SWIR2": 2190
    }

    for band, value in spectral_bands.items():
        fig.add_vline(
            x=value,
            line_dash="dash",
            annotation_text=band,
            annotation_position="top"
        )

    fig.add_vrect(x0=400, x1=700, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_vrect(x0=700, x1=1300, fillcolor="red", opacity=0.08, line_width=0)
    fig.add_vrect(x0=1300, x1=2500, fillcolor="blue", opacity=0.08, line_width=0)

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=24)
        ),
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        template="plotly_white",
        hovermode="x unified",
        height=720,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            rangeslider=dict(visible=True)
        )
    )

    return fig


def create_derivative_plot(wavelength, reflectance):
    first_derivative = np.gradient(reflectance, wavelength)
    second_derivative = np.gradient(first_derivative, wavelength)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=first_derivative,
            mode="lines",
            name="First Derivative"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=second_derivative,
            mode="lines",
            name="Second Derivative"
        )
    )

    fig.update_layout(
        title="📈 Derivative Spectroscopy",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Derivative Value",
        template="plotly_white",
        hovermode="x unified",
        height=650
    )

    return fig


# ============================================================
# DASHBOARD PAGE
# ============================================================

if main_menu == "🏠 Dashboard":

    st.markdown("<div class='main-title'>🌿 ASD Hyperspectral Intelligence Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>FieldSpec 4 Spectral Analysis Platform</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Sensor", "ASD FieldSpec 4")
    col2.metric("Range", "350–2500 nm")
    col3.metric("Mode", "Reflectance")
    col4.metric("Status", "Ready")

    st.markdown("---")

    st.markdown(
        """
        <div class='info-box'>
        This dashboard is designed for ASD FieldSpec 4 hyperspectral data analysis.
        It supports single ASD analysis, multi-ASD comparison, derivative spectroscopy,
        spectral heatmap generation, vegetation stress indices, and AI-ready dataset creation.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SINGLE SPECTRAL ANALYSIS
# ============================================================

elif main_menu == "📊 Spectral Analysis":

    st.title("📊 ASD Spectral Signature Analysis")

    uploaded_file = st.file_uploader(
        "📂 Upload ASD File",
        type=["asd"],
        key="single_asd_upload"
    )

    if uploaded_file is not None:

        try:
            temp_path = save_temp_file(uploaded_file)

            wavelength, reflectance, df = load_asd_file(temp_path)

            smooth_reflectance = apply_smoothing_filter(reflectance)

            st.success("✅ ASD File Loaded Successfully")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Bands", len(wavelength))
            col2.metric("Min Wavelength", f"{wavelength.min():.2f} nm")
            col3.metric("Max Wavelength", f"{wavelength.max():.2f} nm")
            col4.metric("Mean Reflectance", f"{np.mean(reflectance):.4f}")

            fig = create_spectral_plot(
                wavelength,
                reflectance,
                smooth_reflectance,
                "🌿 ASD Spectral Reflectance Signature"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🌱 Vegetation / Spectral Indices")

            index_df = calculate_indices(wavelength, smooth_reflectance)

            st.dataframe(index_df, use_container_width=True)

            st.subheader("📊 Spectral Statistics")

            stats_df = pd.DataFrame({
                "Parameter": [
                    "Minimum Reflectance",
                    "Maximum Reflectance",
                    "Mean Reflectance",
                    "Standard Deviation",
                    "Median Reflectance"
                ],
                "Value": [
                    np.min(reflectance),
                    np.max(reflectance),
                    np.mean(reflectance),
                    np.std(reflectance),
                    np.median(reflectance)
                ]
            })

            st.dataframe(stats_df, use_container_width=True)

            st.subheader("📄 Spectral Data Preview")

            st.dataframe(df.head(50), use_container_width=True)

            csv = df.to_csv(index=False)

            st.download_button(
                "📥 Download Spectral CSV",
                data=csv,
                file_name="asd_spectral_data.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error Loading ASD File: {e}")

    else:
        st.info("📂 Please upload ASD spectral file.")


# ============================================================
# MULTI ASD COMPARISON
# ============================================================

elif main_menu == "📂 Multi-ASD Comparison":

    st.title("📂 Multi-ASD Spectral Comparison")

    multi_files = st.file_uploader(
        "📂 Upload Multiple ASD Files",
        type=["asd"],
        accept_multiple_files=True,
        key="multi_asd_upload"
    )

    if multi_files:

        comparison_fig = go.Figure()
        summary_list = []
        heatmap_data = []

        for uploaded_asd in multi_files:

            try:
                temp_path = save_temp_file(uploaded_asd)

                wavelength, reflectance, df = load_asd_file(temp_path)

                smooth_reflectance = apply_smoothing_filter(reflectance)

                comparison_fig.add_trace(
                    go.Scatter(
                        x=wavelength,
                        y=smooth_reflectance,
                        mode="lines",
                        name=uploaded_asd.name
                    )
                )

                summary_list.append({
                    "File": uploaded_asd.name,
                    "Bands": len(wavelength),
                    "Mean Reflectance": np.mean(reflectance),
                    "Max Reflectance": np.max(reflectance),
                    "Min Reflectance": np.min(reflectance),
                    "Std Reflectance": np.std(reflectance)
                })

                heatmap_data.append(smooth_reflectance)

            except Exception as e:
                st.error(f"❌ Error Processing {uploaded_asd.name}: {e}")

        comparison_fig.update_layout(
            title=dict(
                text="🌿 Multi-ASD Spectral Comparison",
                x=0.5,
                font=dict(size=24)
            ),
            xaxis_title="Wavelength (nm)",
            yaxis_title="Reflectance",
            template="plotly_white",
            hovermode="x unified",
            height=750
        )

        st.plotly_chart(comparison_fig, use_container_width=True)

        st.subheader("📊 Multi-ASD Summary Table")

        summary_df = pd.DataFrame(summary_list)

        st.dataframe(summary_df, use_container_width=True)

        st.download_button(
            "📥 Download Comparison Summary",
            data=summary_df.to_csv(index=False),
            file_name="multi_asd_summary.csv",
            mime="text/csv"
        )

    else:
        st.info("📂 Upload multiple ASD files for comparison.")










# ============================================================
# 🧠 DATASET BUILDER
# ============================================================

elif main_menu == "🧠 Dataset Builder":

    st.title("🧠 ASD Dataset Builder")

    st.info(
        "Upload multiple ASD files and generate AI-ready hyperspectral datasets."
    )

    st.markdown("---")

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    dataset_files = st.file_uploader(

        "📂 Upload ASD Files",

        type=["asd"],

        accept_multiple_files=True,

        key="dataset_builder_upload"
    )

    # ========================================================
    # CLASS NAME
    # ========================================================

    class_name = st.text_input(

        "🏷 Enter Class Name / Disease Name",

        placeholder="Example : Healthy / Diseased / Water Stress"
    )

    st.markdown("---")

    # ========================================================
    # PROCESS BUTTON
    # ========================================================

    if st.button("🚀 Generate AI Dataset"):

        # ====================================================
        # VALIDATION
        # ====================================================

        if not dataset_files:

            st.warning("⚠ Please upload ASD files.")

        elif class_name == "":

            st.warning("⚠ Please enter class name.")

        else:

            progress_bar = st.progress(0)

            status_text = st.empty()

            dataset_rows = []

            # =================================================
            # PROCESS FILES
            # =================================================

            for idx, file in enumerate(dataset_files):

                try:

                    status_text.info(
                        f"📡 Processing : {file.name}"
                    )

                    temp_path = save_temp_file(file)

                    wavelength, reflectance, df = load_asd_file(
                        temp_path
                    )

                    smooth_reflectance = apply_smoothing_filter(
                        reflectance
                    )

                    # =============================================
                    # IMPORTANT BANDS
                    # =============================================

                    def get_band(target_nm):

                        band_idx = np.argmin(
                            np.abs(wavelength - target_nm)
                        )

                        return smooth_reflectance[band_idx]

                    blue = get_band(490)

                    green = get_band(560)

                    red = get_band(665)

                    red_edge = get_band(705)

                    nir = get_band(842)

                    swir1 = get_band(1610)

                    swir2 = get_band(2190)

                    # =============================================
                    # VEGETATION INDICES
                    # =============================================

                    ndvi = (
                        (nir - red)
                        /
                        (nir + red + 1e-10)
                    )

                    ndwi = (
                        (green - nir)
                        /
                        (green + nir + 1e-10)
                    )

                    ndre = (
                        (nir - red_edge)
                        /
                        (nir + red_edge + 1e-10)
                    )

                    msi = (
                        swir1
                        /
                        (nir + 1e-10)
                    )

                    # =============================================
                    # FEATURE ROW
                    # =============================================

                    feature_row = {

                        "File_Name": file.name,

                        "Class": class_name,

                        "Blue": blue,

                        "Green": green,

                        "Red": red,

                        "RedEdge": red_edge,

                        "NIR": nir,

                        "SWIR1": swir1,

                        "SWIR2": swir2,

                        "NDVI": ndvi,

                        "NDWI": ndwi,

                        "NDRE": ndre,

                        "MSI": msi,

                        "Mean_Reflectance":
                            np.mean(smooth_reflectance),

                        "Std_Reflectance":
                            np.std(smooth_reflectance),

                        "Max_Reflectance":
                            np.max(smooth_reflectance),

                        "Min_Reflectance":
                            np.min(smooth_reflectance)
                    }

                    dataset_rows.append(feature_row)

                    # =============================================
                    # UPDATE PROGRESS
                    # =============================================

                    progress = int(
                        ((idx + 1) / len(dataset_files)) * 100
                    )

                    progress_bar.progress(progress)

                except Exception as e:

                    st.error(
                        f"❌ Error Processing {file.name}: {e}"
                    )

            # =================================================
            # CREATE DATAFRAME
            # =================================================

            dataset_df = pd.DataFrame(dataset_rows)

            status_text.success(
                "✅ Dataset Generated Successfully"
            )

            st.markdown("---")

            # =================================================
            # METRICS
            # =================================================

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Files",
                len(dataset_df)
            )

            col2.metric(
                "Features",
                len(dataset_df.columns)
            )

            col3.metric(
                "Class",
                class_name
            )

            col4.metric(
                "Mean NDVI",
                f"{dataset_df['NDVI'].mean():.3f}"
            )

            st.markdown("---")

            # =================================================
            # DATAFRAME
            # =================================================

            st.subheader(
                "📊 AI Ready Spectral Dataset"
            )

            st.dataframe(

                dataset_df,

                use_container_width=True
            )

            # =================================================
            # FEATURE VISUALIZATION
            # =================================================

            st.subheader(
                "📈 Feature Distribution"
            )

            feature_option = st.selectbox(

                "Select Feature",

                [

                    "NDVI",

                    "NDWI",

                    "NDRE",

                    "MSI",

                    "Mean_Reflectance"
                ]
            )

            fig_feature = go.Figure()

            fig_feature.add_trace(

                go.Bar(

                    x=dataset_df["File_Name"],

                    y=dataset_df[feature_option],

                    text=np.round(
                        dataset_df[feature_option],
                        3
                    ),

                    textposition="outside"
                )
            )

            fig_feature.update_layout(

                title=f"{feature_option} Comparison",

                xaxis_title="ASD Files",

                yaxis_title=feature_option,

                template="plotly_white",

                height=550
            )

            st.plotly_chart(

                fig_feature,

                use_container_width=True
            )

            # =================================================
            # DOWNLOAD BUTTON
            # =================================================

            csv = dataset_df.to_csv(
                index=False
            )

            st.download_button(

                "📥 Download AI Dataset CSV",

                data=csv,

                file_name=f"{class_name}_ASD_Dataset.csv",

                mime="text/csv"
            )

            st.success(
                "✅ AI-ready hyperspectral dataset exported successfully."
            )











# ============================================================
# DERIVATIVE SPECTROSCOPY
# ============================================================

elif main_menu == "📈 Derivative Spectroscopy":

    st.title("📈 Derivative Spectroscopy")

    uploaded_file = st.file_uploader(
        "📂 Upload ASD File",
        type=["asd"],
        key="derivative_upload"
    )

    if uploaded_file is not None:

        temp_path = save_temp_file(uploaded_file)

        wavelength, reflectance, df = load_asd_file(temp_path)

        smooth_reflectance = apply_smoothing_filter(reflectance)

        fig = create_derivative_plot(wavelength, smooth_reflectance)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📂 Upload ASD file for derivative spectroscopy.")


# ============================================================
# SPECTRAL HEATMAP
# ============================================================

elif main_menu == "🔥 Spectral Heatmap":

    st.title("🔥 Spectral Heatmap")

    heatmap_files = st.file_uploader(
        "📂 Upload Multiple ASD Files",
        type=["asd"],
        accept_multiple_files=True,
        key="heatmap_upload"
    )

    if heatmap_files:

        spectra_matrix = []
        file_names = []

        for file in heatmap_files:

            temp_path = save_temp_file(file)

            wavelength, reflectance, df = load_asd_file(temp_path)

            smooth_reflectance = apply_smoothing_filter(reflectance)

            spectra_matrix.append(smooth_reflectance)
            file_names.append(file.name)

        fig = go.Figure(
            data=go.Heatmap(
                z=spectra_matrix,
                x=wavelength,
                y=file_names,
                colorscale="Viridis"
            )
        )

        fig.update_layout(
            title="🔥 Spectral Reflectance Heatmap",
            xaxis_title="Wavelength (nm)",
            yaxis_title="ASD Files",
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📂 Upload multiple ASD files to generate heatmap.")


# ============================================================
# VEGETATION STRESS
# ============================================================

elif main_menu == "🌿 Vegetation Stress":

    st.title("🌿 Vegetation Stress Analysis")

    uploaded_file = st.file_uploader(
        "📂 Upload ASD File",
        type=["asd"],
        key="vegetation_upload"
    )

    if uploaded_file is not None:

        temp_path = save_temp_file(uploaded_file)

        wavelength, reflectance, df = load_asd_file(temp_path)

        smooth_reflectance = apply_smoothing_filter(reflectance)

        index_df = calculate_indices(wavelength, smooth_reflectance)

        st.subheader("🌱 Vegetation Stress Indices")

        st.dataframe(index_df, use_container_width=True)

        ndvi_value = index_df[index_df["Index"] == "NDVI"]["Value"].values[0]

        if ndvi_value > 0.5:
            st.success("✅ Plant Condition: Healthy Vegetation")
        elif ndvi_value > 0.2:
            st.warning("⚠ Plant Condition: Moderate Stress")
        else:
            st.error("❌ Plant Condition: High Stress / Diseased Condition")


# ============================================================
# 📚 ADVANCED DOCUMENTATION
# ============================================================

elif main_menu == "📚 Documentation":

    st.title("📚 ASD Spectral Documentation")

    st.markdown("---")

    st.header("🌿 Project Information")

    st.info(
        """
        This project is developed for advanced analysis of ASD FieldSpec 4 hyperspectral data.
        The system helps in spectral signature analysis, vegetation stress detection,
        soil analysis, mineral identification, water stress study, and machine learning
        dataset generation using ASD reflectance data.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Device", "ASD FieldSpec 4")
    col2.metric("Spectral Range", "350–2500 nm")
    col3.metric("Data Type", "Reflectance")
    col4.metric("Application", "Remote Sensing")

    st.markdown("---")

    st.header("🛰 Device Information: ASD FieldSpec 4")

    st.markdown(
        """
        The ASD FieldSpec 4 is a high-resolution spectroradiometer used to collect
        reflectance signatures from vegetation, soil, minerals, water, crops, and
        disease-affected plant samples.

        It records reflected electromagnetic energy from the visible, near-infrared,
        and shortwave infrared regions.
        """
    )

    device_df = pd.DataFrame({
        "Parameter": [
            "Instrument Name",
            "Spectral Range",
            "Main Regions",
            "Output Data",
            "Common Use",
            "Data Format"
        ],
        "Description": [
            "ASD FieldSpec 4",
            "350 nm to 2500 nm",
            "VIS, Red Edge, NIR, SWIR1, SWIR2",
            "Wavelength vs Reflectance",
            "Vegetation, soil, mineral, crop and disease analysis",
            ".asd spectral file"
        ]
    })

    st.dataframe(device_df, use_container_width=True)

    st.markdown("---")

    st.header("📊 Number of Bands and Spectral Regions")

    st.markdown(
        """
        ASD FieldSpec 4 captures a continuous hyperspectral curve.
        Instead of only a few bands like multispectral satellites, hyperspectral data
        contains hundreds or thousands of narrow spectral bands.
        """
    )

    bands_df = pd.DataFrame({
        "Spectral Region": [
            "Visible Blue",
            "Visible Green",
            "Visible Red",
            "Red Edge",
            "Near Infrared",
            "Shortwave Infrared 1",
            "Shortwave Infrared 2"
        ],
        "Approx. Wavelength": [
            "450–495 nm",
            "520–570 nm",
            "620–700 nm",
            "680–750 nm",
            "700–1300 nm",
            "1300–1900 nm",
            "1900–2500 nm"
        ],
        "Main Use": [
            "Pigment and water clarity study",
            "Healthy vegetation reflectance",
            "Chlorophyll absorption",
            "Vegetation stress detection",
            "Leaf structure and biomass",
            "Moisture and water stress",
            "Mineral and soil moisture analysis"
        ]
    })

    st.dataframe(bands_df, use_container_width=True)

    st.markdown("---")

    st.header("🌈 How Important Bands Work")

    band_work_df = pd.DataFrame({
        "Band": [
            "Blue",
            "Green",
            "Red",
            "Red Edge",
            "NIR",
            "SWIR1",
            "SWIR2"
        ],
        "Wavelength Used": [
            "490 nm",
            "560 nm",
            "665 nm",
            "705 nm",
            "842 nm",
            "1610 nm",
            "2190 nm"
        ],
        "Scientific Meaning": [
            "Sensitive to pigments, atmosphere, and water quality",
            "High reflectance in healthy green vegetation",
            "Strong chlorophyll absorption region",
            "Transition zone between red and NIR; useful for stress detection",
            "High reflectance due to internal leaf structure",
            "Sensitive to leaf and soil moisture",
            "Useful for mineral, dry matter, and soil composition analysis"
        ]
    })

    st.dataframe(band_work_df, use_container_width=True)

    st.markdown("---")

    st.header("🌱 Subject-wise Wavelength Applications")

    subject_df = pd.DataFrame({
        "Subject / Application": [
            "Vegetation Health",
            "Plant Disease Detection",
            "Water Stress",
            "Soil Moisture",
            "Mineral Mapping",
            "Crop Biomass",
            "Chlorophyll Content",
            "Dry Matter Analysis"
        ],
        "Useful Bands / Regions": [
            "Red, Red Edge, NIR",
            "Blue, Green, Red, Red Edge, NIR, SWIR",
            "NIR, SWIR1, SWIR2",
            "SWIR1, SWIR2",
            "SWIR1, SWIR2",
            "NIR, Red Edge",
            "Green, Red, Red Edge",
            "SWIR1, SWIR2"
        ],
        "Reason": [
            "NDVI and NIR response indicate plant vigor",
            "Disease changes pigment, moisture, and reflectance pattern",
            "Water absorption is strong in NIR and SWIR regions",
            "Soil moisture strongly affects SWIR reflectance",
            "Minerals have unique absorption features in SWIR",
            "NIR reflectance increases with internal leaf structure",
            "Chlorophyll absorbs red and reflects green",
            "Dry matter affects shortwave infrared response"
        ]
    })

    st.dataframe(subject_df, use_container_width=True)

    st.markdown("---")

    st.header("🧮 Common Spectral Indices")

    indices_df = pd.DataFrame({
        "Index": [
            "NDVI",
            "NDWI",
            "NDRE",
            "MSI"
        ],
        "Formula": [
            "(NIR - Red) / (NIR + Red)",
            "(Green - NIR) / (Green + NIR)",
            "(NIR - RedEdge) / (NIR + RedEdge)",
            "SWIR1 / NIR"
        ],
        "Purpose": [
            "Vegetation health analysis",
            "Water content analysis",
            "Early vegetation stress detection",
            "Moisture stress analysis"
        ]
    })

    st.dataframe(indices_df, use_container_width=True)

    st.markdown("---")

    st.header("🔄 Step-by-Step Workflow")

    workflow_df = pd.DataFrame({
        "Step": [
            "Step 1",
            "Step 2",
            "Step 3",
            "Step 4",
            "Step 5",
            "Step 6",
            "Step 7"
        ],
        "Process": [
            "Upload ASD file",
            "Read wavelength and reflectance values",
            "Apply Savitzky-Golay smoothing",
            "Visualize spectral signature",
            "Extract important bands",
            "Calculate spectral indices",
            "Generate AI-ready dataset"
        ],
        "Output": [
            ".asd file loaded",
            "Spectral dataframe",
            "Noise-reduced spectral curve",
            "Interactive Plotly graph",
            "Blue, Green, Red, NIR, SWIR values",
            "NDVI, NDWI, NDRE, MSI",
            "CSV dataset for ML model"
        ]
    })

    st.dataframe(workflow_df, use_container_width=True)

    st.markdown("---")

    st.success(
        """
        ✅ This documentation section explains project information, ASD device details,
        number of bands, band working principles, subject-wise wavelength applications,
        spectral indices, and complete research workflow.
        """
    )

# ============================================================
# Profile Detail
# ============================================================

elif main_menu == "👨‍💻 Profile Detail":

    st.title(" PROFILE ")

    st.markdown("---")

    # ========================================================
    # INTRODUCTION
    # ========================================================

    st.info(
        """
        Welcome to the ASD Intelligence Platform.

        This system is developed for advanced hyperspectral analysis using
        ASD FieldSpec 4 spectral data for vegetation monitoring, soil analysis,
        mineral mapping, disease detection, and AI-based dataset generation.
        """
    )

    




    # ========================================================
    # 👨‍💻 DEVELOPER PROFILE SECTION
    # ========================================================

    import os
    import glob

    st.markdown("---")

    st.header("👨‍💻 My Information")

    # ========================================================
    # PROFILE + DETAILS
    # ========================================================

    col1, col2 = st.columns([1, 3])

    with col1:

        profile_path = "images/18.jpg"

        if os.path.exists(profile_path):

            st.image(
                profile_path,
                width=260,
                caption="Dr. Jaypalsing Kayte"
            )

        else:

            st.warning("⚠ Profile image not found.")

    with col2:

        st.markdown(
            """
            ## Dr. Jaypalsing N. Kayte

            🎓 PhD in Computer Science & Information Technology

            💼 Senior Software Engineer | AI Technical Team Lead

            🌍 Research Areas:
            - Artificial Intelligence
            - Machine Learning
            - Remote Sensing & GIS
            - Hyperspectral Imaging
            - Deep Learning
            - Spectral Data Analytics
            - Computer Vision

            📧 Email:
            jaypalsing10@gmail.com

            🏢 Organization: Dr. BAM University 
            [ DoITNew Lab ] 

            🌐 Project:
            ASD Hyperspectral Intelligence Platform

            🚀 Developed using:
            Python + Streamlit + Plotly + ASD Spectral Analytics
            """
        )

    st.markdown("---")

    # ========================================================
    # 🎥 CENTER VIDEO SECTION
    # ========================================================

    st.header("🎥 Research Field Video")

    video_path = "images/video.MOV"

    if os.path.exists(video_path):

        video_col1, video_col2, video_col3 = st.columns([1, 3, 1])

        with video_col2:

            st.video(video_path)

    else:

        st.warning("⚠ Video file not found.")

    st.markdown("---")

    # ========================================================
    # 🖼 RESEARCH GALLERY
    # ========================================================

    st.header("🖼 Research Field Gallery")

    # ========================================================
    # LOAD ALL IMAGES
    # ========================================================

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]

    image_files = []

    for ext in image_extensions:

        image_files.extend(
            glob.glob(os.path.join("images", ext))
        )

    # ========================================================
    # REMOVE PROFILE IMAGE
    # ========================================================

    image_files = [

        img for img in image_files

        if "14.jpg" not in img
    ]

    # ========================================================
    # DISPLAY IMAGES IN GRID
    # ========================================================

    num_cols = 4

    rows = [

        image_files[i:i + num_cols]

        for i in range(0, len(image_files), num_cols)
    ]

    for row in rows:

        cols = st.columns(num_cols)

        for col, img_path in zip(cols, row):

            with col:

                st.image(
                    img_path,
                    use_container_width=True
                )

                st.caption(
                    os.path.basename(img_path)
                )

    st.markdown("---")

    # ========================================================
    # CONTACT INFORMATION
    # ========================================================

    st.header("📞 Contact Information")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "📧 Email",
        "jaypalsing10@gmail.com"
    )

    c2.metric(
        "🌍 Domain",
        "AI & Remote Sensing"
    )

    c3.metric(
        "💻 Platform",
        "Python + Streamlit"
    )

    st.markdown("---")

    st.success(
        """
        ✅ ASD Intelligence Platform successfully configured
        for hyperspectral AI research and spectral analytics.
        """
    )












    # ========================================================
    # SYSTEM WORKFLOW
    # ========================================================

    st.header("⚙ Step-by-Step User Workflow")

    workflow_df = pd.DataFrame({

        "Step": [

            "Step 1",
            "Step 2",
            "Step 3",
            "Step 4",
            "Step 5",
            "Step 6"
        ],

        "Process": [

            "Upload ASD spectral file",
            "Apply smoothing filter",
            "Visualize spectral signature",
            "Extract important spectral bands",
            "Generate vegetation indices",
            "Export AI-ready dataset"
        ],

        "Output": [

            ".asd file loaded",
            "Noise reduction",
            "Interactive spectral graph",
            "Blue, Red, NIR, SWIR values",
            "NDVI, NDWI, NDRE, MSI",
            "CSV dataset for AI/ML"
        ]
    })

    st.dataframe(

        workflow_df,

        use_container_width=True
    )

    st.markdown("---")

    # ========================================================
    # FEATURES
    # ========================================================

    st.header("🚀 Platform Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.success(
            """
            ✅ Single ASD Analysis

            ✅ Multi-ASD Comparison

            ✅ Spectral Smoothing

            ✅ Derivative Spectroscopy

            ✅ Spectral Heatmap
            """
        )

    with feature_col2:

        st.success(
            """
            ✅ Vegetation Stress Analysis

            ✅ Spectral Indices

            ✅ AI Dataset Builder

            ✅ Plotly Visualization

            ✅ CSV Export Support
            """
        )

    st.markdown("---")

    # ========================================================
    # CONTACT SECTION
    # ========================================================

    st.header("📞 Contact Information")

    contact_col1, contact_col2, contact_col3 = st.columns(3)

    contact_col1.metric(
        "📧 Email",
        "jaypalsing10@gmail.com"
    )

    contact_col2.metric(
        "🌍 Domain",
        "AI & Remote Sensing"
    )

    contact_col3.metric(
        "💻 Platform",
        "Python + Streamlit"
    )

    st.markdown("---")

    st.success(
        """
        ✅ ASD Intelligence Platform is successfully configured
        for advanced hyperspectral research and AI-based analysis.
        """
    )

   