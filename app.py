import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Anomaly Detection System", layout="wide")
st.title("📊 Anomaly Detection Web App")
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'dfs_multi' not in st.session_state:
    st.session_state.dfs_multi = None
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
mode = st.radio("Select Mode:", ["Single File", "Multiple Files (2-3) Comparison"])

def inject_anomalies(data, time_col=None, anomaly_fraction=0.3):
    num_anomalies = int(len(data) * anomaly_fraction)
    anomalies = data.sample(n=num_anomalies).copy()
    if "Total_Blocks" in anomalies.columns:
        anomalies["Total_Blocks"] = np.random.randint(0, 9, num_anomalies)
    if "Difficulty_Mean" in anomalies.columns:
        anomalies["Difficulty_Mean"] *= np.random.uniform(1.9, 2.9, num_anomalies)
    if "Transaction_Sum" in anomalies.columns:
        anomalies["Transaction_Sum"] *= np.random.uniform(1.1, 2.9, num_anomalies)
    if time_col and time_col in anomalies.columns:
        anomalies[time_col] = anomalies[time_col].sample(frac=1).values
    anomalies["Anomaly"] = 1
    data["Anomaly"] = 0
    return pd.concat([data, anomalies], ignore_index=True)

if mode == "Single File":
    uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"], key="single")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview (Original)")
        st.dataframe(df.head())
        st.subheader("🧩 Edit Column Names")
        new_columns = {}
        for col in df.columns:
            new_columns[col] = st.text_input(f"Rename '{col}' to:", col, key=f"single_col_{col}")
        df.rename(columns=new_columns, inplace=True)
        st.session_state.processed_df = df  
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        st.subheader("Updated Dataset Preview (After Rename)")
        st.dataframe(df.head())
        time_col = st.selectbox("Select timestamp column (if available)", [None] + list(df.columns), key="single_time")
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        anomaly_fraction = st.slider("Anomaly Fraction / Contamination:", 0.01, 0.5, 0.3, 0.01)
        anomaly_mode = st.radio("Anomaly Detection Mode:", ["Synthetic Injection (Supervised)", "Unsupervised Detection"])
        if anomaly_mode == "Synthetic Injection (Supervised)":
            df = inject_anomalies(df, time_col, anomaly_fraction)
            st.success(f"✅ {anomaly_fraction*100:.1f}% anomalies injected successfully!")
        else:
            non_features = []
            if time_col:
                non_features.append(time_col)
            X_iso = df.drop(columns=non_features).select_dtypes(include=[np.number])
            if X_iso.empty:
                st.error("No numeric columns for unsupervised detection!")
            else:
                iso = IsolationForest(contamination=anomaly_fraction, random_state=42)
                anomaly_labels = iso.fit_predict(X_iso)
                df["Anomaly"] = np.where(anomaly_labels == -1, 1, 0)  
                st.success(f"✅ Anomalies detected unsupervised with {anomaly_fraction*100:.1f}% contamination!")
        st.session_state.processed_df = df
        st.subheader("📈 Anomaly Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="Anomaly", palette="viridis", ax=ax_bar)
            ax_bar.set_title("Anomaly Count (Bar)")
            st.pyplot(fig_bar)
        
        with col2:
            anomaly_counts = df["Anomaly"].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
            ax_pie.pie(anomaly_counts.values, labels=["Normal", "Anomaly"], autopct='%1.1f%%', colors=['skyblue', 'orange'])
            ax_pie.set_title("Anomaly Proportion (Pie)")
            st.pyplot(fig_pie)
        if st.button("🚀 Train Anomaly Detection Model", key="single_train"):
            if "Anomaly" not in df.columns:
                st.error("No 'Anomaly' column found in the dataset!")
            else:
                non_features = ["Anomaly"]
                if time_col:
                    non_features.append(time_col)
                X = df.drop(columns=non_features).select_dtypes(include=[np.number])
                
                if X.empty:
                    st.error("No numeric columns available for training! Please check your dataset.")
                else:
                    y = df["Anomaly"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, stratify=y, random_state=42
                    )

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"], key="single_model")
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                    else:
                        model = XGBClassifier(
                            random_state=42,
                            eval_metric="logloss",
                            verbosity=0,
                            tree_method='hist'
                        )

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.subheader("📊 Model Results")
                    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))
                    st.text("Confusion Matrix:")
                    st.write(confusion_matrix(y_test, y_pred))
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Dataset with Anomalies",
                        data=csv,
                        file_name="anomaly_dataset.csv",
                        mime="text/csv"
                    )

                    st.success("🎯 Model trained and evaluated successfully!")

    else:
        st.info("👆 Upload a dataset to begin anomaly detection.")

elif mode == "Multiple Files (2-3) Comparison":
    uploaded_files = st.file_uploader("📂 Upload 2-3 CSV datasets for comparison", type=["csv"], 
                                      accept_multiple_files=True, key="multi")
    
    if len(uploaded_files) < 2:
        st.warning("⚠️ Please upload at least 2 files for comparison (up to 3).")
    elif len(uploaded_files) > 3:
        st.error("❌ Maximum 3 files allowed. Processing only first 3.")
        uploaded_files = uploaded_files[:3]
    
    if uploaded_files:
        dfs = []
        file_names = [f.name for f in uploaded_files]
        time_col = None  # Will be set once
        
        for i, uploaded_file in enumerate(uploaded_files):
            df_temp = pd.read_csv(uploaded_file)
            st.subheader(f"Dataset Preview (Original): {file_names[i]}")
            st.dataframe(df_temp.head())
            st.subheader(f"🧩 Edit Column Names for {file_names[i]}")
            new_columns = {}
            for col in df_temp.columns:
                new_columns[col] = st.text_input(f"Rename '{col}' to:", col, key=f"multi_col_{i}_{col}")
            df_temp.rename(columns=new_columns, inplace=True)
            df_temp.columns = df_temp.columns.str.strip().str.replace(" ", "_")
            st.subheader(f"Updated Dataset Preview (After Rename): {file_names[i]}")
            st.dataframe(df_temp.head())
            if i == 0:
                time_col = st.selectbox("Select timestamp column (if available)", [None] + list(df_temp.columns), key="multi_time")
            
            if time_col and time_col in df_temp.columns:
                df_temp[time_col] = pd.to_datetime(df_temp[time_col], errors="coerce")
            df_temp.fillna(df_temp.median(numeric_only=True), inplace=True)
            df_temp.dropna(axis=1, how="all", inplace=True)
            df_temp["Source"] = file_names[i]
            
            if i == 0:
                st.session_state.dfs_multi = [df_temp]
            else:
                st.session_state.dfs_multi.append(df_temp)
            
            dfs.append(df_temp)
        anomaly_fraction = st.slider("Anomaly Fraction / Contamination:", 0.01, 0.5, 0.3, 0.01)
        anomaly_mode = st.radio("Anomaly Detection Mode:", ["Synthetic Injection (Supervised)", "Unsupervised Detection"])
        
        processed_dfs = []
        for i, df_temp in enumerate(dfs):
            if anomaly_mode == "Synthetic Injection (Supervised)":
                df_temp = inject_anomalies(df_temp, time_col, anomaly_fraction)
                st.success(f"✅ {anomaly_fraction*100:.1f}% anomalies injected in {file_names[i]}!")
            else:
                non_features = []
                if time_col:
                    non_features.append(time_col)
                X_iso = df_temp.drop(columns=non_features).select_dtypes(include=[np.number])
                if not X_iso.empty:
                    iso = IsolationForest(contamination=anomaly_fraction, random_state=42 + i)  # Vary per file
                    anomaly_labels = iso.fit_predict(X_iso)
                    df_temp["Anomaly"] = np.where(anomaly_labels == -1, 1, 0)
                    st.success(f"✅ Anomalies detected unsupervised in {file_names[i]} with {anomaly_fraction*100:.1f}% contamination!")
                else:
                    st.error(f"No numeric columns in {file_names[i]} for detection!")
            processed_dfs.append(df_temp)
        
        st.session_state.dfs_multi = processed_dfs
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        st.session_state.combined_df = combined_df
        st.success("✅ All files processed successfully!")
        
        st.subheader("📊 Anomaly Comparison Across Files")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        sns.countplot(data=combined_df, x="Source", hue="Anomaly", palette="Set2", ax=ax_bar)
        ax_bar.set_title("Anomaly Counts by Source (Bar Chart)")
        st.pyplot(fig_bar)
        
        cols = st.columns(len(uploaded_files))
        for i, df_temp in enumerate(processed_dfs):
            with cols[i]:
                anomaly_counts = df_temp["Anomaly"].value_counts()
                fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                colors = ['lightgreen', 'red'] if i % 2 == 0 else ['lightblue', 'purple']
                ax_pie.pie(anomaly_counts.values, labels=["Normal", "Anomaly"], autopct='%1.1f%%', colors=colors)
                ax_pie.set_title(f"Anomaly Proportion: {file_names[i]} (Pie)")
                st.pyplot(fig_pie)
        
        fig_line, ax_line = plt.subplots(figsize=(10, 6))
        for source in combined_df["Source"].unique():
            source_data = combined_df[combined_df["Source"] == source].sort_index()
            ax_line.plot(range(len(source_data)), source_data["Anomaly"].cumsum(), label=source, linewidth=2)
        ax_line.set_title("Cumulative Anomalies by Source (Line Chart)")
        ax_line.legend()
        ax_line.set_xlabel("Index")
        ax_line.set_ylabel("Cumulative Anomalies")
        st.pyplot(fig_line)
        
        if st.button("🚀 Train Model on Combined Data", key="multi_train"):
            if "Anomaly" not in combined_df.columns:
                st.error("No 'Anomaly' column found!")
            else:
                non_features = ["Anomaly", "Source"]
                if time_col:
                    non_features.append(time_col)
                X = combined_df.drop(columns=non_features).select_dtypes(include=[np.number])
                
                if X.empty:
                    st.error("No numeric columns available for training!")
                else:
                    y = combined_df["Anomaly"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, stratify=y, random_state=42
                    )

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"], key="multi_model")
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                    else:
                        model = XGBClassifier(
                            random_state=42,
                            eval_metric="logloss",
                            verbosity=0,
                            tree_method='hist'
                        )

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.subheader("📊 Combined Model Results")
                    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))
                    st.text("Confusion Matrix:")
                    st.write(confusion_matrix(y_test, y_pred))
                    csv = combined_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Combined Dataset with Anomalies",
                        data=csv,
                        file_name="combined_anomaly_dataset.csv",
                        mime="text/csv"
                    )

                    st.success("🎯 Combined model trained and evaluated successfully!")
        for i, df_temp in enumerate(processed_dfs):
            csv_temp = df_temp.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Download {file_names[i]} with Anomalies",
                data=csv_temp,
                file_name=f"{file_names[i]}_anomalies.csv",
                mime="text/csv",
                key=f"download_{i}"
            )
else:
    st.info("👆 Select a mode and upload datasets to begin.")