# Pepsi_Sentiment_Analysis

This GitHub repository contains a sentiment analysis tool tailored for evaluating comments across Instagram and TikTok posts, focusing on a specific brand- Pepsi. The tool classifies comments based on emotions into positive, negative, or neutral categories.

## Social Media Sentiment Analyzer

Welcome to the Social Media Sentiment Analyzer repository! This project focuses on evaluating sentiments expressed in comments across Instagram and TikTok posts, specifically targeting a designated brand. The reason why this project was undertaken was because the researcher wanted to know which of the social media platforms had more positive or negative comments. 

### Installation

To get started, make sure you have the necessary libraries installed. You can do this by running:

```bash
!pip install transformers numpy pandas torch scikit-learn emoji
```

### Usage

This sentiment analysis tool follows a structured approach to analyze and classify sentiments expressed in social media comments. Here's a breakdown of the steps involved:

1. **Library Installation**: Install required libraries for data processing and model training.
   
2. **Import Libraries**: Import essential libraries for data manipulation, model loading, and evaluation.

```bash
# Import libraries

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
```

3. **Load Fine-tuned BERT Model**: Utilize a pre-trained BERT model fine-tuned for sequence classification tasks.

```bash
# Load the fine-tuned BERT model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
```

4. **Data Preparation**: Load and preprocess the dataset containing Instagram comments related to the target brand. This includes cleaning the data by converting emojis to words and removing special characters.

```bash
!pip install emoji
import re
from emoji import demojize

df['Comment Text'] = df['Comment Text'].str.lower().apply(demojize)
df['Comment Text'] = df['Comment Text'].apply(lambda x: re.sub(r'[:,@_\W]', ' ', x))
```

5. **Tokenization and Preprocessing**: Tokenize and preprocess the comments using the BERT tokenizer to prepare them for sentiment analysis.

```bash
# Tokenize and preprocess your comments for sentiment analysis
encoded_data = tokenizer(df['Comment Text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)
```   

6. **Sentiment Analysis**: Extract BERT embeddings for the preprocessed comments and predict sentiment labels (Positive, Negative, Neutral) using the fine-tuned model.

```bash
# Extract BERT embeddings for your comments
with torch.no_grad():
    outputs = fine_tuned_model(**encoded_data)
    logits = outputs.logits  # Use logits instead of pooler_output
    predicted_labels = torch.argmax(logits, dim=1) + 1  # Adding 1 because the model predicts labels from 0 to 4


# Map numeric sentiment labels to categories (Negative, Neutral, Positive)
sentiment_mapping = {1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'}
df['Sentiment_Class'] = pd.Series(predicted_labels.numpy()).map(sentiment_mapping)

```
    
7. **KNN Classification**: Employ a K-nearest neighbors (KNN) classifier to further classify the comments based on their BERT embeddings and predicted sentiment labels.

```bash
# Specify the feature (X) and target (y) variables for training
# The least populated class in y has only 3 members, was less than n_splits=5 so we adjusted the value of n_splits to 3.

X = logits.numpy()  # Assuming logits contains the BERT embeddings

# Use these predicted sentiment labels as target (y) for KNN
y = df['Sentiment_Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

knn_classifier = KNeighborsClassifier()

grid_search = GridSearchCV(knn_classifier, param_grid, cv=3)

grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']

print("Best value for k:", best_k)

# Create the KNN classifier with the optimal k value
knn_classifier = KNeighborsClassifier(n_neighbors=best_k)

# Train the KNN classifier on the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn_classifier.predict(X_test)

# Evaluate the performance of the KNN model
print(classification_report(y_test, y_pred))
```

8. **Evaluation**: Evaluate the performance of the KNN classifier using standard classification metrics such as precision, recall, and F1-score.


## Key Considerations

- **Model Selection**: The use of a fine-tuned BERT model ensures robust performance in sentiment analysis tasks, but alternative models can be explored based on specific requirements and constraints.

- **Data Cleaning**: The preprocessing steps ensure that the comments are properly formatted and devoid of noise, enhancing the effectiveness of sentiment analysis.

- **Parameter Tuning**: The choice of parameters, particularly the number of neighbors (k) in the KNN classifier, significantly impacts model performance. Grid search is employed to find the optimal k value, but further tuning may be necessary.

- **Documentation**: Thorough documentation and comments within the code promote clarity and reproducibility, facilitating collaborative development and understanding.

- **Scalability**: While the current implementation is suitable for small to moderate-sized datasets, scalability considerations should be taken into account for larger datasets.

### Contributions

Contributions to this repository are highly encouraged! Whether it's bug fixes, feature enhancements, or optimizations, your contributions are valuable in improving the effectiveness and usability of the sentiment analysis tool.

### License

This project is licensed under the [MIT License](LICENSE), allowing for flexibility in usage and modification. Feel free to adapt the code to suit your specific needs and requirements.


Start exploring the Social Media Sentiment Analyzer now and unlock valuable insights into the sentiments expressed on Instagram and TikTok. Your feedback and contributions are vital in shaping the future of this project. Let's work together to make social media sentiment analysis more accessible and insightful!
