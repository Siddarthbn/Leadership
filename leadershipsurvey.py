import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import linprog

# Survey questions for each variable
questions = {
    "Planning": [
        "How effectively does your organisation assess its resource needs to align with strategic goals and high-performance outcomes?",
        "How agile is the organisation’s planning process in addressing new challenges and seizing opportunities in a rapidly evolving business environment?"
    ],
    "Capital": [
        "How proactively does your organisation monitor and respond to technological, market, and competitive changes in its industry?",
        "In what ways does your organisation adapt to shifting customer demands, regulatory changes, and stakeholder expectations?"
    ],
    "Resources": [
        "How well does your organisation align financial, technological, and human resources to meet high-performance objectives?",
        "How effectively does the organisation leverage its resources to foster innovation and achieve operational excellence?"
    ],
    "Governance": [
        "How robust and transparent are the governance structures in ensuring accountability and ethical compliance?",
        "To what extent do governance practices align operational decisions with the organisation’s strategic vision?"
    ]
}

# User inputs
st.title("VCLARIFI Leadership")

# Add VTARA logo
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
logo_path = os.path.join(desktop_path, "VTARA.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
else:
    st.sidebar.warning("Logo not found on Desktop.")

# Initialize session state for responses
if 'responses' not in st.session_state:
    st.session_state['responses'] = {}

# Input for user's name
user_name = st.text_input("Please enter your name:")

# Input for team selection
team = st.selectbox("Select your team:", ["IT Department", "Management", "Finance"])

# Input for user's place
place = st.selectbox("Please select your place:", ["Bengaluru", "Canberra"])

# Add submit button to confirm user's name, team, and place
if st.button("Confirm Details"):
    if user_name and team and place:
        st.session_state['responses'][user_name] = {
            "Name": user_name,
            "Team": team,
            "Place": place
        }

if user_name in st.session_state['responses']:
    # Display survey questions after confirming user's details
    st.subheader("Survey Questions")
    user_responses = st.session_state['responses'][user_name]

    # Likert scale options
    likert_options = [
        "1: Not at all", "2: To a very little extent", "3: To a little extent",
        "4: To a moderate extent", "5: To a fairly large extent",
        "6: To a great extent", "7: To a very great extent"
    ]

    for variable, qs in questions.items():
        st.header(variable)
        responses = []
        for q in qs:
            response = st.radio(q, options=likert_options, index=3, key=f"{user_name}_{q}")
            numeric_value = int(response.split(":")[0])
            responses.append(numeric_value)
        user_responses[variable] = np.mean(responses)

    if st.button("Submit Survey"):
        # Save current response
        if 'responses_list' not in st.session_state:
            st.session_state['responses_list'] = []
        st.session_state['responses_list'].append(user_responses)

        # Display user input results
        st.subheader("User Input Results")
        st.write(f"Name: {user_responses['Name']}")
        st.write(f"Team: {user_responses['Team']}")
        st.write(f"Place: {user_responses['Place']}")
        for variable, score in user_responses.items():
            if variable not in ['Name', 'Team', 'Place']:
                st.write(f"{variable} - Average Score: {score:.2f}")

        # Visualize user input scores
        variables = [var for var in user_responses.keys() if var not in ['Name', 'Team', 'Place']]
        mean_scores = [user_responses[var] for var in variables]
        benchmarks = [5.5] * len(variables)  # Set benchmark as 5.5
        fig, ax = plt.subplots()
        ax.bar(variables, mean_scores, color='skyblue', label='User Scores')
        ax.axhline(y=5.5, color='red', linestyle='--', label='Benchmark (5.5)')
        ax.set_ylim(0, 7)  # Set y-axis to Likert scale range
        ax.set_title('Leadership Elements Scores with Benchmarks')
        ax.set_ylabel('Scores (1-7)')
        ax.legend()
        st.pyplot(fig)

        # Define file path to save Excel file on Desktop
        file_path = os.path.join(desktop_path, "survey_responses.xlsx")

        # Check if the Excel file already exists and load previous data
        if os.path.exists(file_path):
            previous_df = pd.read_excel(file_path, index_col=0)
        else:
            previous_df = pd.DataFrame()

        # Append current responses to previous data
        new_df = pd.DataFrame(st.session_state['responses_list'], index=[f"{user_responses['Name']}_{i+1}" for i in range(len(st.session_state['responses_list']))])
        combined_df = pd.concat([previous_df, new_df])

        # Store results in Excel file
        try:
            combined_df.to_excel(file_path)
            st.success(f"Survey results have been saved successfully to {file_path}.")
        except PermissionError:
            alternative_path = os.path.join(os.path.expanduser("~"), "Documents", "survey_responses.xlsx")
            combined_df.to_excel(alternative_path)
            st.success(f"Survey results have been saved successfully to {alternative_path}.")

        # Load planning scores for the current user
        user_responses_df = combined_df[combined_df.index.str.startswith(user_name)]
        planning_scores = user_responses_df['Planning'].values

        # Fit ARIMA model for forecasting
        if len(planning_scores) >= 3:  # Ensure at least 3 data points for ARIMA model
            try:
                model = ARIMA(planning_scores, order=(1, 1, 0))
                model_fit = model.fit()
                # Forecast future scores
                forecast = model_fit.forecast(steps=1)
                st.subheader("Forecasted Planning Score")
                st.write(f"Forecasted Planning Score for {user_name}: {forecast[0]:.2f}")
            except Exception as e:
                st.error(f"An error occurred while fitting the ARIMA model: {e}")
        else:
            remaining_surveys = 3 - len(planning_scores)
            st.warning(f"{user_name} needs {remaining_surveys} more survey(s) to forecast the Planning score.")

        # Linear Programming for Capital
        st.subheader("Optimized Resource Allocation for Capital")
        c = [-1, -1]  # Coefficients for optimization (maximize scores)
        A = [[1, 2], [3, 1]]  # Example resource constraints
        b = [7, 8]  # Limits for the constraints
        res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
        if res.success:
            st.write(f"Optimized Resource Allocation for Capital: {res.x}")
        else:
            st.error("Linear programming optimization failed.")

        # Compare teams across all variables
        st.subheader("Team Performance Comparison")
        if not previous_df.empty:
            numeric_columns = previous_df.select_dtypes(include=[np.number]).columns
            team_comparison_df = previous_df.groupby('Team')[numeric_columns].mean()
            st.write("Average Scores by Team:")
            st.dataframe(team_comparison_df)
            best_team = team_comparison_df.mean(axis=1).idxmax()
            st.write(f"Best Performing Team: {best_team}")

            # Compare same team across different locations
            st.subheader("Team Performance Comparison by Place")
            for team in team_comparison_df.index:
                team_place_comparison_df = previous_df[previous_df['Team'] == team].groupby('Place')[numeric_columns].mean()
                st.write(f"Average Scores for {team} by Place:")
                st.dataframe(team_place_comparison_df)
                best_place_for_team = team_place_comparison_df.mean(axis=1).idxmax()
                st.write(f"Best Performing Place for {team}: {best_place_for_team}")
        else:
            st.write("No previous survey data available for comparison.")

        # Compare places across all variables
        st.subheader("Place Performance Comparison")
        if not previous_df.empty:
            place_comparison_df = previous_df.groupby('Place')[numeric_columns].mean()
            st.write("Average Scores by Place:")
            st.dataframe(place_comparison_df)
            best_place = place_comparison_df.mean(axis=1).idxmax()  # Corrected here
            st.write(f"Best Performing Place: {best_place}")
        else:
            st.write("No previous survey data available for comparison.")
