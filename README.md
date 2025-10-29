# 🛍️ Customer Segmentation & Targeted Marketing Analysis

### 📌 Overview
This project applies **Unsupervised Machine Learning** using **K-Means clustering** to segment customers based on their **Age**, **Annual Income**, and **Spending Score**.  
A fully interactive **Streamlit Dashboard** is developed to visualize insights and help businesses create **targeted marketing strategies** for each customer segment.

Streamlit AppLink : https://customer-segmentation-app-mhclppg3kxkmj3ma6mxjdk.streamlit.app/

## 🎯 Objectives
- Understand customer shopping behavior
- Segment customers into meaningful groups
- Analyze cluster characteristics and purchasing power
- Support targeted marketing strategies with data insights
- Demonstrate practical Data Analytics + ML + Business skills

---

## 📂 Tech Stack
| Category | Tools |
|---------|------|
| Language | Python |
| ML Model | K-Means Clustering |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Interactive App | Streamlit |
| Version Control | Git & GitHub |

---

## 📊 Dataset Details
- **Mall Customers Dataset**
- Source: Public dataset (Kaggle)
- Features used:
  - CustomerID (removed during analysis)
  - Age
  - Gender (encoded)
  - Annual Income (k$)
  - Spending Score (1–100)

---

## 🏷️ Clusters Identified (Business Interpretation)

| Cluster | Description | Behavior | Marketing Strategy |
|--------|-------------|----------|-------------------|
| Cluster 0 | Budget-Conscious Youth | Low income, moderate spending | Coupons, discounts, loyalty offers |
| Cluster 1 | High-Value Loyal Customers | High income, high spending | Premium memberships, exclusive perks |
| Cluster 2 | Low-Value Risk Group | Low income, low spending | Awareness campaigns, bundle offers |
| Cluster 3 | Potential High Buyers | High income, low spending | Brand engagement, personalized ads |
| Cluster 4 | Average Customers | Mid income & spending | Balanced marketing approach |

➡️ These insights enable **smarter customer retention and acquisition** planning.

---

## 📈 Visualizations Included
- Elbow Method to determine optimal clusters
- 2D scatter plots with cluster separation
- Spending habits visualization
- Demographic distribution graphs
- Cluster-based business insights

---

## 🖥️ Streamlit Dashboard Features
✅ Upload or use default dataset  
✅ Automated clustering  
✅ Interactive charts  
✅ Cluster interpretation  
✅ User-friendly layout  

To run:

```sh
streamlit run app.py
🧠 Key Learnings

Data preprocessing & encoding

Feature scaling improves clustering

K-Means parameter tuning using Elbow Method

Turning models into business decisions

Deploying interactive analytical dashboards

🚀 Project Structure
Customer-Segmentation-Project/
│── app.py
│── dataset.csv
│── model.pkl
│── README.md
└── visualizations/


(We will add graph images here later ✅)

📌 Future Enhancements

Deploy online using Streamlit Cloud

Add cluster-based prediction for new users

Add more features like Annual Spending / Purchase History

Include Marketing ROI optimization metrics

👤 Author

Somyashree Nayak
📍 India
