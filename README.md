This comprehensive README.md provides a clear overview of the "QuLab: Interactive MA Time Series Analysis" Streamlit application, detailing its purpose, features, setup, and structure.

---

# QuLab: Interactive Moving Average (MA) Time Series Analysis

![Streamlit App Screenshot](https://raw.githubusercontent.com/your-username/your-repo-name/main/docs/screenshot.png)
*(Note: Replace with an actual screenshot of your running application)*

## 📄 Project Description

This Streamlit application, **QuLab**, serves as an interactive educational tool for exploring Moving Average (MA) time series models and their Autocorrelation Function (ACF). Developed as part of a QuantUniversity lab project, it provides a user-friendly interface to:

*   **Generate synthetic MA data**: By interactively adjusting model parameters such as MA order (q=1 or 2), theta coefficients, and white noise standard deviation.
*   **Visualize time series**: Display the generated MA data over time.
*   **Analyze Autocorrelation Function (ACF)**: Calculate and plot the ACF for the generated series, complete with confidence intervals, to illustrate the characteristic 'cut-off' property of MA processes.

The application aims to enhance understanding of fundamental time series concepts through hands-on experimentation.

## ✨ Features

*   **Interactive MA Model Generation**:
    *   Choose between MA(1) and MA(2) models.
    *   Adjust $\theta_1$ and $\theta_2$ coefficients dynamically.
    *   Control the standard deviation of the white noise ($\epsilon$).
    *   Set the number of data points (samples) for the time series.
*   **Time Series Visualization**: Plot the generated MA time series using an interactive Plotly chart.
*   **ACF Calculation & Plotting**:
    *   Automatically compute the Autocorrelation Function (ACF).
    *   Display ACF values with confidence intervals to assess statistical significance.
    *   Customize the number of lags for the ACF plot.
    *   Adjust the significance level ($\alpha$) for confidence intervals.
*   **Mathematical Explanations**: Provides LaTeX-formatted equations for MA models and ACF definitions directly within the app, aiding in conceptual understanding.
*   **Responsive UI**: Built with Streamlit for a clean, intuitive, and responsive user experience.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.7+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *(Note: Replace `your-username/your-repo-name` with your actual repository details)*

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

    **`requirements.txt` content:**
    ```
    streamlit>=1.0.0
    pandas>=1.0.0
    numpy>=1.20.0
    plotly>=5.0.0
    statsmodels>=0.13.0
    scipy>=1.5.0
    ```

## 🛠️ Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**
    Open your web browser and navigate to the URL displayed in your terminal (usually `http://localhost:8501`).

3.  **Interact with the application:**
    *   Use the **sidebar** on the left to adjust parameters for the MA model (Order, Theta Coefficients, White Noise Standard Deviation, Number of Samples) and ACF plot settings (Number of Lags, Significance Level).
    *   Observe the "Generated Time Series" plot and the "Autocorrelation Function Plot" in the main content area, which update interactively with your parameter changes.
    *   Explore the mathematical explanations provided within the application.

## 📁 Project Structure

```
.
├── app.py
├── application_pages/
│   ├── __init__.py
│   ├── page1.py
│   └── page2.py  # Placeholder for future pages
├── requirements.txt
└── README.md
```

*   `app.py`: The main Streamlit entry point. It sets up the page configuration, displays the header, and handles navigation to different application pages.
*   `application_pages/`: A directory containing the individual pages (modules) of the Streamlit application.
    *   `page1.py`: Contains all the logic, functions (`generate_ma_data`, `calculate_acf`, `plot_acf`), and Streamlit UI elements for the "MA Model and ACF" analysis.
    *   `page2.py`: A placeholder for additional pages or functionalities that might be added in the future.
*   `requirements.txt`: Lists all Python package dependencies required to run the application.
*   `README.md`: This file, providing an overview of the project.

## 💻 Technology Stack

*   **Framework**: [Streamlit](https://streamlit.io/) (for building the web application)
*   **Core Language**: [Python](https://www.python.org/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
*   **Numerical Operations**: [NumPy](https://numpy.org/)
*   **Statistical Functions**: [SciPy](https://scipy.org/) (for `norm.ppf`)
*   **Time Series Analysis**: [Statsmodels](https://www.statsmodels.org/stable/index.html) (for `acf` calculation)
*   **Plotting**: [Plotly](https://plotly.com/python/) (for interactive visualizations)

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes and commit them (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for details (if you plan to add one, otherwise state it's proprietary/educational as per the app footer).

*(Note: The application footer states "The purpose of this demonstration is solely for educational use and illustration. Any reproduction of this demonstration requires prior written consent from QuantUniversity." This implies a more restrictive license than MIT. For a lab project, if open-sourcing, you might choose MIT. If it's strictly for QuantUniversity, you might omit or clarify the license to reflect their terms.)*

## 📞 Contact

This lab was generated using the QuCreate platform. For inquiries or collaboration related to QuantUniversity projects, please visit:

*   **Website**: [QuantUniversity](https://www.quantuniversity.com)

---
**Disclaimer from the Application:**
"The purpose of this demonstration is solely for educational use and illustration. Any reproduction of this demonstration requires prior written consent from QuantUniversity. This lab was generated using the QuCreate platform. QuCreate relies on AI models for generating code, which may contain inaccuracies or errors."
---