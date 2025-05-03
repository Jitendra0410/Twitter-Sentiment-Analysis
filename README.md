
# 🐦 Twitter Sentiment Analysis with Visualization

This project performs **Twitter sentiment analysis** using machine learning on pre-labeled tweet datasets. It classifies tweets into **Positive**, **Negative**, **Neutral**, and **Irrelevant** categories. The project also visualizes tweet statistics and model performance with interactive and static graphs.

## 📁 Files Used

- `twitter_training.csv`: Training dataset containing tweets and their sentiment labels.
- `twitter_validation.csv`: Validation dataset used for testing model performance.

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab

## ⚙️ Steps Performed

### 1. Data Loading & Preprocessing
- Read both training and validation datasets
- Cleaned null entries
- Encoded sentiment labels using `LabelEncoder`
- Sampled for speed (10,000 training and full validation data)

### 2. Exploratory Data Analysis (EDA)
- Checked tweet length distribution across sentiments
- Identified class imbalance
- Generated statistical summaries (mean, median, etc.)

### 3. Model Training
- Used **Multinomial Naive Bayes** classifier
- Trained on TF-IDF transformed tweet content
- Evaluated using validation set

### 4. Visualization
Minimum of 5 types of graphs:
- 📊 **Bar Chart**: Count of tweets by sentiment
- 📈 **Line Plot**: Tweet length over sample index
- 🥧 **Pie Chart**: Sentiment distribution
- 📦 **Box Plot**: Tweet length by sentiment
- 📉 **Radar Chart**: Precision, Recall, F1-score by class

### 5. Evaluation
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

## 📌 Key Results

- **Accuracy Achieved**: ~84% (varies slightly by run)
- Most tweets classified as Neutral
- Neutral tweets are shorter on average than Positive/Negative ones

## 📷 Sample Output Graphs

- ![Pie Chart of Sentiments](images/pie_chart.png)
- ![Boxplot of Tweet Lengths](images/boxplot.png)
- ![Radar Plot of Metrics](images/radar_chart.png)
- ![Confusion Matrix](images/confusion_matrix.png)

## 📚 How to Run

Open the project in [Google Colab](https://colab.research.google.com/):

```python
# Load datasets
cols = ['tweet_id', 'entity', 'sentiment', 'content']
train_df = pd.read_csv('twitter_training.csv', names=cols)
val_df = pd.read_csv('twitter_validation.csv', names=cols)
