import streamlit as st
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

st.set_page_config(page_title="AutoML Pipeline", layout="wide")

def main():
    st.title("🤖 Automated Machine Learning Pipeline")

    st.sidebar.title("📂 Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    st.sidebar.title("⚙️ Configuration")
    problem_type = st.sidebar.selectbox("Select Problem Type", ["classification", "regression", "clustering"])
    target_column_name = st.sidebar.text_input("Target Column Name")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.expander("🔍 Preview Uploaded Data"):
            st.dataframe(df)

        if st.sidebar.button("🚀 Run Analysis"):
            with st.spinner("Starting the pipeline..."):

                try:
                    # Data Ingestion
                    st.info("📥 Data Ingestion in progress...")
                    data_ingestion = DataIngestion()
                    train_data, test_data, eda_report_path = data_ingestion.initiate_data_ingestion(df)
                    st.success("✅ Data Ingestion Completed")

                    # EDA Report
                    with st.expander("📊 EDA Report"):
                        with open(eda_report_path, "rb") as f:
                            st.download_button("Download EDA Report", f, file_name="eda_report.html")

                    # Data Transformation
                    st.info("🔄 Data Transformation in progress...")
                    data_transformation = DataTransformation()
                    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                        train_data, test_data, problem_type, target_column_name)
                    st.success("✅ Data Transformation Completed")

                    # Model Training
                    st.info("🤖 Model Training in progress...")
                    model_trainer = ModelTrainer()
                    best_model_name, best_model_score, model_report = model_trainer.initiate_model_trainer(
                        train_arr, test_arr, problem_type)

                    st.success("✅ Model Training Completed")

                    st.subheader("📈 Model Results")
                    tab1, tab2 = st.tabs(["📊 Model Comparison", "🏆 Best Model"])

                    with tab1:
                        if problem_type in ['regression', 'classification']:
                            comparison_df = pd.DataFrame.from_dict(
                                model_report, orient='index', columns=['Train Score', 'Test Score'])
                        else:
                            comparison_df = pd.DataFrame.from_dict(
                                model_report, orient='index', columns=['Silhouette Score'])

                        st.dataframe(comparison_df)

                    with tab2:
                        st.markdown(f"**Best Model**: `{best_model_name}`")
                        st.markdown(f"**Best Score**: `{best_model_score}`")

                except Exception as e:
                    st.error(f"🚨 Error occurred: {str(e)}")
                    logging.error(f"Pipeline Error: {str(e)}")

    else:
        st.warning("👆 Please upload a CSV file to start.")

if __name__ == "__main__":
    main()
