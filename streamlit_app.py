import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.let_it_rain import rain
import codecs
import streamlit.components.v1 as components
import streamlit_shadcn_ui as ui

import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import pickle
import statsmodels.api as sm
import math
from io import StringIO
import requests


from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


# Load the dataset
df = pd.read_csv("Admission_Predict_Ver1.1.csv")

# Set the page configuration for our Streamlit app
st.set_page_config(
    page_title="Graduate University Admission ",
    layout="centered",
    page_icon="graduation.png", 
)

# Add a sidebar with a title and a radio button for page navigation
st.sidebar.title("GradAdmissions 🎓")
page = st.sidebar.radio(
    "Select Page",
    ("Introduction 👩‍💼", "Visualization 📊", "Prediction 📣")
)

# Display the selected page content based on the user's choice
if page == "Introduction 👩‍💼":
    # Loading Animation
    with st.spinner('Loading page...'):
        # Set the title of the page in rainbow colors
        rainbow_title = """
        <h1 style='text-align: center; font-size: 3.0em; font-weight: bold;'>
        <span style='color: red;'>Graduate</span>
        <span style='color: orange;'>University</span>
        <span style='color: violet;'>Admission</span>
        <span style='color: blue;'>Chance</span>
        <span style='color: indigo;'>Predictor</span>
        </h1>
        """
        st.markdown(rainbow_title, unsafe_allow_html=True)

        # Display the image in the center of the page
        map = Image.open("USAUniversityMap.png")
        # Resize the image to the desired size
        map = map.resize((2500, 1500))
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            st.image(map)
        
        # Set the subtitle and a rainbow divider of the page 
        st.markdown(
            """
            <h3 style='text-align: center;'>Supporting Applicants through Past data and LR Model</h3>
            <div style='height: 4px; 
                        margin: 0 auto 20px auto; 
                        width: 60%%; 
                        background: linear-gradient(90deg, red, orange, yellow, green, blue, indigo, violet); 
                        border-radius: 2px;'>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create a horizontal option menu for navigation
        selected = option_menu(
        menu_title = None,
        options = ["Welcome","Overview","Exploration"],
        icons = ["award", "trophy-fill", "mortarboard"],
        default_index = 0,
        orientation = "horizontal",
        )

        # Display the selected page content based on the user's choice
        if selected == "Welcome":

            st.balloons()

            # Welcome Section
            st.header("💡 Team")
            st.markdown("""
            Welcome to our app! 
            This app is designed by a team of three undergraduate students.
            Our motivation to create this app is to assist prospective graduate students (including ourselves 🙌) in their journey towards higher education. 
            Here is a little bit about us:
            """)
            team_members = {
                "Yazhen Li": "yzlfk087@gmail.com",
                "Christina Chen": "cc8192@nyu.edu",
                "Shirley Shi": "js12861@nyu.edu",
            }
            for name, email in team_members.items():
                st.write(f"- {name} ({email})")

            # Objective Section
            st.header("🎯 Objective")
            st.markdown("""
            The goal of this app is to:
            - Provides insights into the factors that influence admission decisions.
            - Deploy the app in a user-friendly interface for real-time predictions.
            - Build a robust linear regression model that can accurately analyze the relationship between the chance of admission and other variables.
            """)

            # Quick Statistics Section
            st.header("📌 Quick Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Students", df.shape[0])
            with col2:
                st.metric("Average Chance of Admit", round(df['Chance of Admit '].mean(), 2))
            with col3:
                st.metric("Number of Features", df.shape[1])
       
            # st.success("Dive into the data, discover insights, and keep learning .")

        # Display the selected page content based on the user's choice
        elif selected == "Overview":

            # Foreword Section
            st.header("⚙️ Dataset Overview")
            st.markdown("""
            Here is a brief overview of the dataset we used in this app:
            """)

            # Data types
            st.write("### Features Types")
            dtypes = df.dtypes
            dtype_details = {}
            for dtype in dtypes.unique():
                columns = dtypes[dtypes == dtype].index.tolist()
                dtype_details[str(dtype)] = {
                    "Columns": ", ".join(columns),
                    "Count": len(columns)
                }
            dtype_df = pd.DataFrame(dtype_details).T.reset_index()
            dtype_df.columns = ['Data Type', 'Columns', 'Count']
            st.write(dtype_df)

            # Data Description
            field_options = df.columns.tolist()
            selected_field = st.selectbox("Select a field to view its description:", field_options)
            st.write(f"### Description of {selected_field}")
            if selected_field == "Serial No.":
                st.code(["Serial No. : Serial Number"])
            elif selected_field == "SOP":
                st.code(["SOP (0 - 5): Statement of Purpose"])
            elif selected_field == "LOR ":
                st.code(["LOR (0 - 5): Letter of Recommendation Strength"])
            elif selected_field == "CGPA":
                st.code(["CGPA (0 - 10): Culmulative Undergraduate GPA"])
            elif selected_field == "GRE Score":
                st.code(["GRE Score (260 - 340): Graduate Record Examinations Score"])
            elif selected_field == "TOEFL Score":
                st.code(["TOEFL Score (0 - 120): Test of English as a Foreign Language Score"])
            elif selected_field == "University Rating":
                st.code(["University Rating (0 - 5)"])
            elif selected_field == "Research":
                st.code(["Research (either 0 or 1): Research Experience"])
            elif selected_field == "Chance of Admit ":
                st.code(["Chance of Admit (ranging from 0 to 1): Chance of Admission"])

            # Custom CSS to reduce spacing between Features Description Table and Statistical Summary
            st.markdown(
                """
                <style>
                .ag-theme-streamlit {
                    margin-bottom: 0px !important;
                }
                .main {
                    padding-top: 0rem !important;
                    padding-bottom: 0rem !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            # Feature Information
            feature_info = {
                "Column Name": [
                    "Serial No.", "GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR ", "CGPA", "Research",
                    "Chance of Admit "
                ],
                "Description": [
                    "virtualized and unique identifier for each applicant",
                    "a standardized test score that assesses a applicant's verbal reasoning, quantitative reasoning, and analytical writing",
                    "a standardized test score that measures a applicant's English language proficiency",
                    "a measure of the reputation and quality of the university from which the applicant graduated",
                    "a personal essay that outlines applicant's academic passion, professional goals, and motivations",
                    "a measure of the quality of recommendation letters provided by referees",
                    "a measure of a applicant's academic performance during their undergraduate studies",
                    "indicates whether the applicant has prior research experience (1) or not (0)",
                    "the probability of being admitted to a graduate program",
                ]
            }
            # Create a DataFrame for feature information
            feature_df = pd.DataFrame(feature_info)
            grid_options = GridOptionsBuilder.from_dataframe(feature_df)
            grid_options.configure_default_column(width=200)
            grid_options.configure_columns(["Column Name"], width=250)
            grid_options.configure_columns(["Description"], width=450)
            grid_options.configure_columns(["Description"], autoSize=True)
            grid_options.configure_grid_options(domLayout='autoHeight', enableRangeSelection=True, enableSorting=True,
                                                enableFilter=True, pagination=True, paginationPageSize=9,
                                                suppressHorizontalScroll=True, rowHeight=35, headerHeight=35)
            grid_options.configure_column("Description", cellStyle={'white-space': 'normal'})
            grid_options.configure_column("Column Name", cellStyle={'textAlign': 'center'})
            grid_options.configure_column("Column Name", headerClass="header-style")
            grid_options.configure_column("Description", headerClass="header-style")
            grid_options.configure_column("Column Name", cellStyle={'backgroundColor': '#dee2ff'})
            grid_options.configure_column("Description", cellStyle={'backgroundColor': '#e9ecff'})
            # Features Description Table
            st.write("### Features Description Table")
            AgGrid(feature_df, gridOptions=grid_options.build(), fit_columns_on_grid_load=True)

            # Basic statistics
            st.write("### Statistical Summary")
            st.write(df.describe())

            # Optional deeper dive of Missing Values
            st.write("### Missing Values")
            if st.checkbox("Check Missing Values"):
                missing = df.isnull().sum()
                missing = missing[missing > 0]
                if missing.empty:
                    st.success("No missing values detected!")
                else:
                    st.error("Missing values detected!")
                    st.dataframe(missing)
            
            # Data Preview
            st.write("### Data Preview: 10 rows")
            view_option = st.radio("View from:", ("Top", "Bottom"))
            if view_option == "Top":
                st.dataframe(df.head(10))
            else:
                st.dataframe(df.tail(10))
            
            st.write("### Automated Report")
            if st.button("Generate an Automated Report:"):
                st.balloons()
                profile = ProfileReport(df, title="University Graduate Admission Report", explorative=True, minimal=True)
                st_profile_report(profile)
                export = profile.to_html()
                st.download_button(
                    label="📥 Download full Report",
                    data=export,
                    file_name="university_graduate_admission_report.html",
                    mime='text/html',
                )

        # Display the selected page content based on the user's choice
        elif selected == "Exploration":

            # Foreword Section
            st.header("🔎 Dataset Exploration")
            st.markdown("""
            Here is a closer exploration of the dataset we used in this app:
            """)

            # Data Display
            st.write("### 📚 Data Display")
            rows = st.slider("Select a number of rows to display", 5,10) 
            filtered_df = dataframe_explorer(df, case=False)
            st.dataframe(filtered_df.head(rows), use_container_width=True)

            # Optional deeper dive of Feature Distributions
            st.write("### 📂 Feature Distributions")
            if st.checkbox("Show Feature Distributions", key="feature_dist_checkbox"):
                df2 = df.drop(["Serial No."], axis=1)

                numeric_cols = df2.select_dtypes(include=['int64', 'float64']).columns.tolist()

                selected_feature = st.selectbox("Select a feature to explore:", numeric_cols)

                fig, ax = plt.subplots()
                ax.hist(df2[selected_feature].dropna(), bins=20, edgecolor='black')
                ax.set_title(f'Distribution of {selected_feature}')
                ax.set_xlabel(selected_feature)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

            # Correlation Matrix
            st.write("### 🔗 Correlation Matrix")
            df2 = df.drop(["Serial No."], axis=1)
            selected_columns = st.multiselect(
            "Select columns to include in the correlation matrix:",
            options=df2.select_dtypes(include=['float64', 'int64']).columns.tolist(),
            default=df2.select_dtypes(include=['float64', 'int64']).columns.tolist())
            if len(selected_columns) >= 2:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(df2[selected_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Please select at least two columns to display the correlation matrix.")

            st.write("### ✍🏻 Correlation Analysis with Chance of Admit")
            if st.checkbox("Show Feature Distributions", key = "feature_dist_checkbox2"):
                st.write(" - The correlation with 'Chance of Admit' is particularly important as it helps us understand which features are most influential in determining the likelihood of admission." \
                "            Here are some key observations from the correlation matrix:" )
                st.write(" - CGPA has **a very strong positive correlation** with Chance of Admit (0.88), indicating that academic performance is a significant factor in admission decisions.")
                st.write(" - GRE Score has **a strong positive correlation** with Chance of Admit (0.81), indicating that higher GRE scores are associated with better admission chances.")   
                st.write(" - TOEFL Score also shows **a rather positive correlation** with Chance of Admit (0.79), suggesting that better English proficiency is linked to higher admission chances.")
                st.write(" - University Rating, SOP and LOR also show moderate positive correlations with Chance of Admit (0.69, 0.68, and 0.65 respectively), suggesting that strong personal statements and recommendation letters are important for admission.")
                st.write(" - Research experience has a less positive correlation with Chance of Admit (0.55), indicating that prior research experience is beneficial for applicants.")

            st.write("### 🎒 Features Relationships")
            df2 = df.drop(["Serial No."], axis=1)
            # Add color schemes reflecting the U.S.
            list_columns = df2.columns
            values = st.multiselect("Select two variables to compare:", list_columns, ["GRE Score", "Chance of Admit "], max_selections = 2)
            if len(values) == 2:
                # Show line chart between selected features
                st.line_chart(df, x=values[0], y=values[1])
                # Show bar chart between selected features
                st.bar_chart(df, x=values[0], y=values[1])
            else:
                st.info("Please select exactly two variables.")



# Display the selected page content based on the user's choice":
elif page == "Visualization 📊":
    # Loading Animation
    with st.spinner('Loading page...'):
        st.balloons()
        st.markdown(
            """
            <h1 style='text-align: center; font-size: 3.0em; font-weight: bold'>Data Visualization</h1>
            <div style='height: 4px; 
                        margin: 0 auto 20px auto; 
                        width: 60%%; 
                        background: linear-gradient(90deg, red, orange, yellow, green, blue, indigo, violet); 
                        border-radius: 2px;'>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Embed Looker Studio report using iframe
        looker_studio_url = "https://lookerstudio.google.com/embed/reporting/5fb89108-a8b4-4b27-b0ea-ebaba3091216/page/KQuNF"
        components.iframe(looker_studio_url, height=600, width=1000)

        # Custom CSS for the rainbow button
        st.markdown(
            """
            <style>
            .rainbow-button {
                background: linear-gradient(90deg, red, orange, violet, indigo, yellow);
                color: white !important;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
                border: none;
            }
            .rainbow-button:hover {
                opacity: 0.8;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # Layout with columns
        column1, column2, column3 = st.columns([1, 1, 1])
        with column1:
            st.write("")
        with column2:
            # Custom rainbow button using HTML
            looker_studio_url = "https://lookerstudio.google.com/embed/reporting/5fb89108-a8b4-4b27-b0ea-ebaba3091216/page/KQuNF"
            st.markdown(
            f'''
            <a href="{looker_studio_url}" target="_blank" class="rainbow-button">
                <b><span style="color: #fff;">👉🏻 Go To Looker Studio 🌈</span></b>
            </a>
            ''',
            unsafe_allow_html=True
            )
        with column3:
            st.write("")



elif page == "Prediction 📣":
    # Loading Animation
    with st.spinner('Loading page...'):
        # Set the title of the page in rainbow colors
        st.markdown(
            """
            <h1 style='text-align: center; font-size: 3.0em; font-weight: bold'>Linear Regression Prediction</h1>
            <div style='height: 4px; 
                        margin: 0 auto 20px auto; 
                        width: 60%%; 
                        background: linear-gradient(90deg, red, orange, yellow, green, blue, indigo, violet); 
                        border-radius: 2px;'>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        X=df[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]
        y=df['Chance of Admit ']
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.2)
        model = sklearn.linear_model.LinearRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        # Allow users to upload their own dataset
        st.write("### 🧐 Input Features for Your Prediction")
        CGPA=st.number_input("CGPA (Culmulative Undergraduate GPA: 0 - 10)", min_value = 0.0, max_value = 10.0, value = 9.0)
        gre = st.slider("GRE Score (260 - 340)", min_value = 260, max_value = 340, value = 337, step = 1)
        toefl = st.slider("TOEFL Score (0 - 120)", min_value = 0, max_value = 120, value = 112, step = 1)
        univ_rating = st.selectbox("University Rating", options = [1, 2, 3, 4, 5], index = 2)
        sop = st.selectbox("SOP (Statement of Purpose: 0 - 5)", options = [0, 0.5, 1, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5], index = 9)
        LOR = st.selectbox("LOR (Letter of Recommendation Strength: 0 - 5)", options = [0, 0.5, 1, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5], index = 10)
        research = st.radio("Research Experience", options = ["No", "Yes"], index = 1)

        st.sidebar.write("**Your Inputs:**")
        st.sidebar.write(f"CGPA: {CGPA}")
        st.sidebar.write(f"GRE: {gre}")
        st.sidebar.write(f"TOEFL: {toefl}")
        st.sidebar.write(f"University Rating: {univ_rating}")
        st.sidebar.write(f"SOP: {sop}")
        st.sidebar.write(f"LOR: {LOR}")
        st.sidebar.write(f"Research: {research}")

        research_binary = 1 if research == "Yes" else 0

        if st.button("Predict Chance of Admission"):
            input_features=np.array([[gre,toefl,univ_rating,sop,LOR,CGPA,research_binary]])
            predictions= model.predict(input_features)
            chance = round(predictions[0]*100, 2)
            # st.success(f"Your estimated chance of admission is: {predictions[0]*100:.2f}%")
            if chance > 85:
                st.balloons()
                st.success(f"Your estimated chance of admission is: {chance:.2f}%.")
                st.markdown(
                    """
                    <span style='
                        font-size: 1.3em;
                        font-weight: bold;
                        background: linear-gradient(90deg, red, orange, gold, green, blue, indigo, violet);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        display: inline-block;
                    '>
                    Congrats🎉 You have an excellent chance of being admitted to graduate universities!
                    </span>
                    """,
                    unsafe_allow_html=True
                )
            elif 70 <= chance <= 85:
                rain(emoji="🚀", font_size=90, falling_speed=2, animation_length="1")
                st.markdown(
                    f"""
                    <div style='
                        background-color: #fff3cd;
                        color: #856404;
                        padding: 1em;
                        border-radius: 6px;
                        font-size: 1em;
                        margin-top: -2.0rem;   /* Adjust this value to reduce space */
                    '>
                    Your estimated chance of admission is: {chance:.2f}%.<br>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<span style='font-size:1.2em; font-weight:bold; color:#856404;'>Hang on and keep working hard🚀 You are almost there!</span>",
                    unsafe_allow_html=True
                )
            else:
                st.snow()
                st.error(f"Your estimated chance of admission is: {chance:.2f}%.")
                st.markdown(
                    "<span style='font-size:1.2em; font-weight:bold; color:#FF4B4B;'>Fight on💪 Keep improving and you'll get closer to your goal!</span>",
                    unsafe_allow_html=True
                )

        st.subheader("Model Evaluation Metrics (on Test Data)")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Show the evaluation metrics
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**R-squared (R²):** {r2:.2f}")

        # Draw the Predicted vs. Actual Chance of Admit scatter plot
        st.subheader("Predicted vs. Actual Chance of Admit")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(),y_test.max()],
                [y_test.min(),y_test.max() ],"--r", linewidth=2)
        ax.set_xlabel("Actual Chance of Admit")
        ax.set_ylabel("Predicted Chance of Admit")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)