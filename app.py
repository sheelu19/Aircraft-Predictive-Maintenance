import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Aircraft Predictive Maintenance", layout="wide")

preprocessor = joblib.load("preprocessing_pipeline.pkl")
model = joblib.load("best_rf_model.pkl")

required_features = ['op_setting1', 'op_setting2', 's2', 's3', 's4', 's7', 's8',
                     's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predictive Maintenance", "Visualization", "RUL Prediction"])

if page == "Predictive Maintenance":
    st.title("Predictive Maintenance Overview")
    try:
        with open("predictive_maintenance_info.txt", "r", encoding="utf-8") as file:
            content = file.read()
        st.markdown(content)
    except FileNotFoundError:
        st.warning("The file 'predictive_maintenance_info.txt' was not found.")

elif page == "Visualization":
    st.title("Aircraft Sensor Data Visualizations")

    np.random.seed(42)
    n_samples = 500
    synthetic_data = pd.DataFrame({
        'op_setting1': np.random.normal(50, 10, n_samples),
        'op_setting2': np.random.normal(100, 20, n_samples),
        's2': np.random.normal(600, 50, n_samples),
        's3': np.random.normal(800, 60, n_samples),
        's4': np.random.normal(1200, 80, n_samples),
        's7': np.random.normal(250, 25, n_samples),
        's8': np.random.normal(450, 40, n_samples),
        's9': np.random.normal(300, 35, n_samples),
        's11': np.random.normal(90, 15, n_samples),
        's12': np.random.normal(70, 10, n_samples),
        's13': np.random.normal(50, 8, n_samples),
        's14': np.random.normal(400, 30, n_samples),
        's15': np.random.normal(150, 15, n_samples),
        's17': np.random.normal(90, 9, n_samples),
        's20': np.random.normal(300, 40, n_samples),
        's21': np.random.normal(500, 45, n_samples),
        'RUL': np.random.randint(10, 300, n_samples)
    })

    compact_size = (4, 3)

    st.subheader("1. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(synthetic_data.corr(), cmap="coolwarm", ax=ax, cbar=False)
    st.pyplot(fig)

    st.subheader("2. Distribution of Sensor s3")
    fig, ax = plt.subplots(figsize=compact_size)
    sns.histplot(synthetic_data['s3'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("3. Scatter Plot: op_setting1 vs RUL")
    fig, ax = plt.subplots(figsize=compact_size)
    sns.scatterplot(x='op_setting1', y='RUL', data=synthetic_data, ax=ax, s=10)
    st.pyplot(fig)

    st.subheader("4. Box Plot: s11")
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.boxplot(y=synthetic_data['s11'], ax=ax)
    st.pyplot(fig)

    st.subheader("5. Line Plot: Sensor s8 over Time")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=synthetic_data['s8'], ax=ax)
    st.pyplot(fig)

    st.subheader("6. KDE of s14")
    fig, ax = plt.subplots(figsize=compact_size)
    sns.kdeplot(synthetic_data['s14'], fill=True, ax=ax)
    st.pyplot(fig)

    st.subheader("7. Pairplot: Key Sensors (s2, s3, s4, s9)")
    fig = sns.pairplot(synthetic_data[['s2', 's3', 's4', 's9']], height=1.3)
    st.pyplot(fig)

    st.subheader("8. Bar Chart: Average s7 by RUL Group")
    synthetic_data['RUL_group'] = pd.cut(synthetic_data['RUL'], bins=5)
    avg_s7 = synthetic_data.groupby('RUL_group')['s7'].mean()
    fig, ax = plt.subplots(figsize=compact_size)
    avg_s7.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("9. Violin Plot: s17")
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.violinplot(y=synthetic_data['s17'], ax=ax)
    st.pyplot(fig)

    st.subheader("10. Area Plot: op_setting2 and s15")
    fig, ax = plt.subplots(figsize=(6, 3))
    synthetic_data[['op_setting2', 's15']].head(50).plot.area(ax=ax, alpha=0.4)
    st.pyplot(fig)

    st.subheader("11. Sensor Variance Heatmap")
    fig, ax = plt.subplots(figsize=(5, 1.5))
    sensor_variances = synthetic_data.select_dtypes(include=[np.number]).var().sort_values(ascending=False)
    sns.heatmap(sensor_variances.to_frame().T, cmap="Blues", ax=ax, cbar=False)
    st.pyplot(fig)

    st.subheader("12. RUL Trend")
    fig, ax = plt.subplots(figsize=(6, 2.5))
    synthetic_data['RUL'].plot(ax=ax, color='green')
    ax.set_ylabel("RUL")
    st.pyplot(fig)

    st.subheader("13. Correlation: Settings vs RUL")
    fig, ax = plt.subplots(figsize=compact_size)
    corr = synthetic_data[['op_setting1', 'op_setting2', 'RUL']].corr()['RUL'].drop('RUL')
    corr.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("14. Outlier Detection: s13")
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.boxplot(y=synthetic_data['s13'], ax=ax, color='red')
    st.pyplot(fig)

    st.subheader("15. Cumulative Distribution of RUL")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.ecdfplot(data=synthetic_data, x='RUL', ax=ax)
    st.pyplot(fig)



elif page == "RUL Prediction":
    st.title("Aircraft Predictive Maintenance")
    st.markdown("Choose input method:")
    input_mode = st.radio("Select input type", ["Manual Entry", "Upload CSV"])

    if input_mode == "Manual Entry":
        st.markdown("Enter the input data for a single prediction:")
        input_data = {}
        for feature in required_features:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)
        input_df = pd.DataFrame([input_data])
        if st.button("Predict RUL (Manual)"):
            try:
                input_prepared = preprocessor.transform(input_df)
                prediction = model.predict(input_prepared)
                st.success(f"Predicted Remaining Useful Life (RUL): {prediction[0]:.2f} cycles")
            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.markdown("Upload a CSV file containing sensor data")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(df.head())
                missing_cols = [col for col in required_features if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    if st.button("Predict RUL (Batch)"):
                        input_prepared = preprocessor.transform(df[required_features])
                        predictions = model.predict(input_prepared)
                        df["Predicted_RUL"] = predictions
                        st.success("RUL predictions completed")
                        st.markdown("Predicted RUL values:")
                        st.dataframe(df[["Predicted_RUL"]].head(20))
                        st.markdown("Full result preview:")
                        st.dataframe(df.head(10))
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Results as CSV", data=csv, file_name="rul_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error reading or processing file: {e}")
