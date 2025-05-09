# ⚽️ Football Player Statistics & Analysis (EPL 2024–2025)

## 📊 Project Overview

This project is a **comprehensive data pipeline and analysis tool** for player statistics in the **2024–2025 English Premier League (EPL)** season. From data collection to insightful visualizations and predictive modeling, this project delivers deep insights into player performance and market value.

### 🎯 Objectives
1. **Data Collection**  
   - Scrape detailed statistics for players who have played more than **90 minutes** — including passing, shooting, progression, defense, and more. <br> 
   - Save to `results.csv`.  
2. **Statistical Analysis**  
   - Identify the **top 3 players** (highest **and** lowest) for each statistic; save to `top_3.txt`.  
   - Calculate the **median** for each statistic.  
   - Calculate the **mean** and **standard deviation** for each statistic **across all players** and **for each team**; save to `results2.csv`.  
   - Plot a **histogram** of each statistic’s distribution for **all players** and **each team** (using `matplotlib.pyplot.hist`).  
   - Identify the **team** with the **highest average** for each statistic.

3. **Data Visualization**  
   - Generate **histograms**, **distributions**, and **comparison charts**.  
   - Apply **K-Means Clustering** and **PCA** to group and classify players.

4. **Transfer Value Estimation**  
   - Scrape and analyze transfer values.  
   - Propose a **predictive model** to estimate player transfer value using **XGBoost**.


---

## 🧰 Tech Stack & Requirements

To run this project, make sure you have the following Python libraries installed:

| Library         | Purpose                                           |
|----------------|---------------------------------------------------|
| `pandas`        | Data manipulation and wrangling                   |
| `numpy`         | Efficient numerical computations                  |
| `matplotlib`    | Data visualization (charts, plots)                |
| `seaborn`       | Statistical data visualization                    |
| `scikit-learn`  | Machine learning (clustering, PCA)                |
| `xgboost`       | Gradient boosting for transfer value estimation   |
| `selenium`      | Automated web scraping                            |
| `beautifulsoup4`| HTML parsing for extracting data                  |
| `joblib`        | Saving/loading trained models                     |
| `re`            | Regular expressions for text filtering            |

### 📦 Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

Start the data collection and analysis process by running the main script:

```bash
python main.py
```

---

## 👤 Author

- **Vũ Thị Thu Duyên**  
- 📧 Email: vuthuduyen.2010@gmail.com 
- 🌐 GitHub: [github.com/ThuDuyenn](https://github.com/ThuDuyenn)