Here's a comprehensive `README.md` file for your Streamlit application lab project, formatted in Markdown.

---

# QuLab: Interactive Time Series Analysis Lab (MA & ARCH Models)

## Project Overview

Welcome to **QuLab**, an interactive Streamlit application designed for an in-depth exploration of fundamental time series models: **Moving Average (MA) processes** and **Autoregressive Conditional Heteroskedasticity (ARCH) models**.

Understanding time series data is critical across various domains, including finance, economics, and environmental science. This application serves as an educational platform to provide hands-on experience and conceptual understanding of these powerful models.

### Key Learning Objectives:

1.  **Moving Average (MA) Processes**: Interact with a simulator to understand how parameters (MA coefficients $\theta$ and white noise standard deviation $\sigma$) influence the shape of an MA time series and its Autocorrelation Function (ACF). Gain intuitive insights into the impact of past error terms on current observations.
2.  **Autoregressive Conditional Heteroskedasticity (ARCH) Models**: Delve into the theory behind models that capture changing volatility, especially prevalent in financial time series. While the interactive component for ARCH models is a future enhancement, this lab introduces their core concepts and mathematical formulations.

This lab provides an intuitive way to visualize complex time series concepts, making them accessible for students and practitioners alike.

## Features

*   **Interactive MA Model Simulation**:
    *   Dynamically adjust the order (`q`) of the MA process (MA(1), MA(2), MA(3)).
    *   Control MA coefficients ($\theta_1, \theta_2, \theta_3$) and white noise standard deviation ($\sigma$).
    *   Adjust the length of the simulated time series (number of years).
*   **Real-time Visualization**:
    *   View the simulated MA time series plot instantly as parameters change.
    *   Observe the Autocorrelation Function (ACF) plot, illustrating the impact of parameters on correlations at different lags, complete with 95% confidence intervals.
*   **ARCH Model Explanations**:
    *   Detailed theoretical introduction to ARCH models, including concepts like heteroskedasticity, conditional variance, and volatility clustering.
    *   Mathematical formulations for the ARCH(1) model and methods for testing ARCH effects.
*   **Clear Navigation**: Easy switching between "Moving Average (MA) Models", "ARCH Models Explained", and "References" sections via a sidebar menu.
*   **Informative Descriptions**: Each section includes comprehensive explanations of the models, their interpretation, and the significance of the displayed visualizations.

### Planned Enhancements (Future Work)

*   Interactive simulations and visualizations for ARCH models.
*   Introduction to GARCH models.
*   Option to load real-world time series data for analysis.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3.8+ installed on your system.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/[your-username]/qu-lab-timeseries.git
    cd qu-lab-timeseries
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**

    Create a `requirements.txt` file in the root directory of your project with the following content:

    ```
    streamlit
    pandas
    numpy
    plotly
    scipy
    statsmodels
    ```

    Then, install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the dependencies are installed, you can run the Streamlit application.

1.  **Run the application:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**
    Your default web browser should automatically open a new tab displaying the QuLab application, usually at `http://localhost:8501`.

3.  **Explore the Lab:**
    *   Use the **sidebar navigation** to switch between "Moving Average (MA) Models", "ARCH Models Explained", and "References".
    *   On the **"Moving Average (MA) Models"** page, use the sliders and radio buttons in the sidebar to adjust the MA order, coefficients, white noise standard deviation, and series length. Observe how the time series and its ACF plot respond in real-time.
    *   On the **"ARCH Models Explained"** page, read through the theoretical concepts of ARCH models.

## Project Structure

The project is organized into modular files for clarity and maintainability:

```
qu-lab-timeseries/
├── app.py                      # Main Streamlit application entry point and navigation.
├── requirements.txt            # List of Python dependencies.
└── application_pages/          # Directory containing individual page modules.
    ├── page1.py                # Contains logic for MA Model simulation and visualization.
    ├── page2.py                # Contains content for ARCH Models explanation.
    └── page3.py                # Contains content for the References page.
```

## Technology Stack

This application is built using the following technologies and libraries:

*   **Python 3.8+**: The core programming language.
*   **Streamlit**: For creating interactive web applications with pure Python.
*   **Pandas**: For data manipulation and analysis, particularly for time series handling.
*   **NumPy**: For numerical computations and array operations.
*   **Plotly**: For creating interactive and publication-quality statistical graphics.
*   **SciPy**: For scientific computing, including statistical functions.
*   **statsmodels**: For statistical modeling, including time series processes (e.g., `arma_generate_sample` and `plot_acf`).

## Contributing

Contributions to enhance QuLab are welcome! If you have suggestions for new features, improvements, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Consider contributing to the planned enhancements, especially adding interactive components for ARCH models!

## License

This project is copyrighted by QuantUniversity. All Rights Reserved. Reproduction or distribution of this demonstration requires prior written consent from QuantUniversity.

## Contact

For any questions, feedback, or inquiries, please contact QuantUniversity.

*   **Website:** [QuantUniversity](https://www.quantuniversity.com/) (Link provided in the application, adjust if a direct contact email is preferred)

---