# Stroke Prediction

This project aims to predict stroke occurrence based on various health and demographic factors. The dataset contains information about patients' demographics, medical history, and lifestyle choices, and is used to predict whether a patient is likely to have a stroke.

## Milestone 2: Data Visualization & Data Encoding

### General Tasks:
1. Framed questions related to stroke prediction and performed data visualization.
2. Plotted graphs to explore various factors influencing stroke occurrence.
3. Made observations from the data visualizations.

### Specific Tasks:
1. Converted `Residence_type` column to `Urban/Rural` (0 = rural, 1 = urban).
2. One-hot encoded `work_type` column into `Never_worked`, `Private`, and `Self-employed`.
3. One-hot encoded `smoking_status` into `formerly smoked`, `never smoked`, and `smokes`.
4. Created a new dataset for machine learning, dropped the original categorical columns (`Residence_type`, `work_type`, `smoking_status`).

## File Structure:
- `milestone_2.py`: Python script for performing data visualization and encoding.
- `data.csv`: The dataset used for analysis.
- `graphs/`: Folder containing visualizations generated from the dataset (e.g., histograms, bar charts, etc.).

## Requirements:
This project requires the following Python libraries:
- `pandas`
- `matplotlib`
- `seaborn`

You can install the required libraries using:

```bash
pip install -r requirements.txt
