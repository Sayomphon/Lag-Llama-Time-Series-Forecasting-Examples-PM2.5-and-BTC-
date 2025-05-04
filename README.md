# Lag-Llama-Time-Series-Forecasting-Examples (PM2.5-and-BTC)

## Overview

This repository contains Jupyter notebooks demonstrating the application of the pre-trained **Lag-Llama** foundation model for time series forecasting. We explore its capabilities on two distinct real-world datasets:

1.  **Environmental Data:** Forecasting daily PM2.5 levels for the Khlong Toei station in Bangkok, Thailand.
2.  **Financial Data:** Forecasting daily closing prices for Bitcoin (BTC).

The notebooks cover the entire workflow, including environment setup, data loading and preprocessing, performing both **zero-shot forecasting** and **fine-tuning** the Lag-Llama model, evaluating the results using the Continuous Ranked Probability Score (CRPS), and visualizing the forecasts.

## Repository Contents

* `Lag_Llama_PM2_5_Forecasting_for_Khlong_Toei_Station.ipynb`: Jupyter notebook for PM2.5 forecasting.
* `Lag-Llama_BTC_close_price_Forecasting.ipynb`: Jupyter notebook for BTC closing price forecasting.
* `Khlong_Toei_PM2.5.csv`: Sample data file for PM2.5 forecasting (or provide instructions on how to obtain it).
* `BTC_price.csv`: Sample data file for BTC price forecasting (or provide instructions on how to obtain it).
* `/Pictures/`: Folder containing result visualization images.
* `README.md`: This file.

## About the Model: Lag-Llama

Lag-Llama is a foundation model for time series forecasting pre-trained on a large corpus of time series data. It leverages a Llama architecture adapted for time series inputs (lags) to perform probabilistic forecasting.

* **Original Repository:** [time-series-foundation-models/lag-llama](https://github.com/time-series-foundation-models/lag-llama)
* **Research Paper:** Rasul, K., et al. (2023). Lag-Llama: Towards Foundation Models for Time Series Forecasting. *arXiv preprint arXiv:2310.08278*. [https://arxiv.org/abs/2310.08278](https://arxiv.org/abs/2310.08278)

## Datasets

1.  **PM2.5 Data (Khlong Toei, Bangkok):**
    * Description: Daily average PM2.5 concentration levels.
    * Source: GISTDA - [https://pm25.gistda.or.th/download](https://pm25.gistda.or.th/download) (Data might require specific selection for the station and date range).
    * File used: `Khlong_Toei_PM2.5.csv` (Please ensure you have this file or download it from the source).
2.  **Bitcoin (BTC) Price Data:**
    * Description: Daily closing prices for Bitcoin (USD).
    * Source: [Specify Source if known, e.g., Yahoo Finance, Kaggle, specific exchange API] or "Provided in `BTC_price.csv`".
    * File used: `BTC_price.csv` (Please ensure you have this file or obtain it from a reliable source).

## Methodology

Both notebooks follow a similar structure:

1.  **Environment Setup:** Clones the official Lag-Llama repository and installs required dependencies using `pip`. Downloads the pre-trained Lag-Llama checkpoint.
2.  **Data Loading & Preprocessing:** Loads the respective dataset (PM2.5 or BTC) using Pandas. Performs preprocessing steps including:
    * Parsing date columns.
    * Handling potential duplicates (by averaging).
    * Resampling to ensure daily frequency.
    * Imputing missing values via linear interpolation.
    * Adding a unique `item_id`.
    * Converting data types.
3.  **Data Splitting:** Divides the data into training and validation sets for the fine-tuning process.
4.  **GluonTS Dataset Creation:** Converts the Pandas DataFrames into `PandasDataset` objects suitable for the Lag-Llama estimator.
5.  **Zero-Shot Forecasting:** Uses the pre-trained Lag-Llama model directly (without fine-tuning) to generate forecasts for different context lengths (e.g., 32, 64, 128).
6.  **Fine-Tuning:** Adapts the pre-trained Lag-Llama model to the specific dataset (PM2.5 or BTC) by training it for a few epochs with a low learning rate. This is also performed for different context lengths. *Note: Fine-tuning requires careful hyperparameter selection, especially for volatile financial data.*
7.  **Evaluation:** Compares the probabilistic forecasts from both zero-shot and fine-tuned models against the actual values using the Continuous Ranked Probability Score (CRPS). Lower CRPS is better.
8.  **Visualization:** Plots the actual time series data against the mean forecasts and prediction intervals from both zero-shot and fine-tuned models for visual comparison.

## Results Summary

The performance (CRPS) comparison between Zero-Shot and Fine-tuned models for different context lengths (`ctx`) is summarized below. Lower CRPS indicates better performance.

**1. PM2.5 Forecasting (Khlong Toei Station)**

| Context | Zero-Shot CRPS | Fine-tuned CRPS | Improvement | % Improvement |
| :------ | :------------- | :-------------- | :---------- | :------------ |
| 32      | 0.3310         | 0.4778          | -0.1468     | -44.36%       |
| 64      | 0.4033         | 0.3529          | 0.0504      | 12.49%        |
| 128     | 0.4170         | 0.6686          | -0.2516     | -60.35%       |

*Observation:* For the PM2.5 dataset with the chosen hyperparameters, fine-tuning generally degraded performance compared to the zero-shot model, except for a context length of 64 which showed slight improvement. This might suggest overfitting or suboptimal hyperparameters for fine-tuning this specific dataset.

**2. BTC Closing Price Forecasting**

| Context | Zero-Shot CRPS | Fine-tuned CRPS | Improvement | % Improvement |
| :------ | :------------- | :-------------- | :---------- | :------------ |
| 32      | 0.0608         | 0.0462          | 0.0145      | 23.92%        |
| 64      | 0.0721         | 0.1213          | -0.0492     | -68.24%       |
| 128     | 0.0772         | 0.0382          | 0.0390      | 50.50%        |

*Observation:* For the BTC dataset, fine-tuning yielded mixed results. It significantly improved performance for context lengths 32 and 128 but drastically worsened it for context length 64. This highlights the sensitivity of fine-tuning, especially on volatile financial data, and suggests that the optimal context length might differ between zero-shot and fine-tuned scenarios.

***

## Result Visualization Examples

Below are sample visualizations comparing the actual data with the Zero-Shot and Fine-tuned forecasts generated by the notebooks.

**PM2.5 Forecast Example (Context Length = 64)**
*Note: This context length showed improvement with fine-tuning in the results table.*

![PM2.5 Forecast Comparison (Context 64)]([./Pictures/PM_64.png](https://github.com/Sayomphon/Lag-Llama-Time-Series-Forecasting-Examples-PM2.5-and-BTC-/blob/main/Pictures/BTC%20128.png))

**BTC Price Forecast Example (Context Length = 128)**
*Note: This context length showed significant improvement with fine-tuning in the results table.*
<p align="center">
  <img src="https://github.com/Sayomphon/Lag-Llama-Time-Series-Forecasting-Examples-PM2.5-and-BTC-/blob/main/Pictures/BTC%20128.png?raw=true" alt="BTC Price Forecast Example (Context Length = 128)">
</p>

***

## Setup & Installation

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/Sayomphon/Lag-Llama-Time-Series-Forecasting-Examples-PM2.5-and-BTC-.git]
    cd Lag-Llama-Time-Series-Forecasting-Examples-PM2.5-and-BTC-
    ```
2.  **Data Files:** Ensure the necessary data files (`Khlong_Toei_PM2.5.csv`, `BTC_price.csv`) are placed in the root directory of this repository, or update the `csv_path` variable within the notebooks accordingly. The sample result images are located in the `/Pictures/` directory.
3.  **Dependencies:** The notebooks handle cloning the Lag-Llama repository and installing its dependencies (`requirements.txt`) automatically when run. Ensure you have `git` installed. Internet access is required for downloading the repository and the pre-trained model checkpoint via `huggingface-cli`.

## Usage

1.  **Environment:** Open the notebooks (`.ipynb` files) in Google Colab (recommended, especially with GPU runtime) or a local Jupyter environment.
2.  **Data Path:** Verify that the `csv_path` variable in **Section 3.1** of each notebook points to the correct location of your data file.
3.  **Run Cells:** Execute the notebook cells sequentially.
    * The first run will take time to clone the Lag-Llama repo, install packages, and download the checkpoint.
    * Fine-tuning steps can be computationally intensive. Using a GPU is recommended.

## Citation

If you use Lag-Llama in your research, please cite the original paper:

```bibtex
@misc{rasul2023lagllama,
      title={Lag-Llama: Towards Foundation Models for Time Series Forecasting},
      author={Kashif Rasul and Arjun Ashok and Andrew Robert Williams and Riham G Badrinarayanan and Hena Ghonia and Medha HEGDE and Richard Tomsett and Bernardo Garcíaळ्यातleda and Ghufran Stanikzai and Toufik CAMALET and Irina Espejo and John Griesbauer and Nicolas Chapados and Cedric VALMARY and Piotr Eliasz and Alexandros SPYRIDONIDIS and Parishad BehnamGhader and Florian Borchert and Valentin Courgeau and David BLOCH and Philippe MOTTIN and Yuriy Nevmyvaka},
      year={2023},
      eprint={2310.08278},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
