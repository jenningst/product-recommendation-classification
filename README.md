<p align='center'>
  <img width="600" src='https://github.com/jenningst/product-recommendation-classification/blob/main/images/burgess-milner-OYYE4g-I5ZQ-unsplash.jpg' alt='Twitter + Phone'>
</p>
<p align='center'>
  Photo by <a href="https://unsplash.com/@burgessbadass?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Burgess Milner</a> on <a href="https://unsplash.com/s/photos/womens-clothing?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
</p>

# Classifying Product Recommendations from Women's e-Commerce Reviews
Consumers have more options than ever when it comes to purchasing products online. Companies like Amazon provide endless optionality for any product and have innovated logistics to the point where same-day delivery to the consumer is possible. Furthermore, companies like Stripe and Shopify have made it increasingly easy for companies to take their brand online, promising the potential to reach customers anywhere. With the drive to take the consumer shopping process online, an exponential amount of data is being generated — click-through data to track the actions of consumers leading up to a purchase, historical purchase behavior data, and consumer product review data are just a few examples of data-mining opportunities in e-commerce. This increase in the availability and volume of customer data presents increased opportunities for businesses to optimize their processes and drive additional profits and growth.

# Motivation
In this project, we attempt to analyze consumer reviews on women’s clothing e-commerce data through natural language processing, sentiment analysis, and feature engineering. Our goal is to combine these techniques to predict the consumer’s recommendation on a given review. Specifically, we want to use classification to predict a recommended label of “not recommended”. Doing so ensures the Customer Success team can engage with as many customers as the model predicts as giving a “not recommended” label. We will evaluate multiple models using different forms of word-representations including bag-of-words, n-grams, and term frequency-inverse document frequency (TF-IDF) on the text data and select the best performing model. We will also use several variations of the feature-set and a variety of algorithms for classification.

# Datasource & Definitions
The dataset utilized for this project was sourced from Kaggle (Figure 1 in References). The data represents a sample of real (albeit anonymized) data from a commercial, women’s clothing E-commerce reviews. The original dataset consists of 10 dependent variables, a binary target variable, and 23,486 instances. Given a contrived problem context, this analysis focuses on 2 key variables — review text and review rating — as well as engineered features derived from these variables. The dependent variable for this project was recommended, a binary variable which indicates whether the consumer recommended the product or not.
To have the data follow the problem context, only the review text, rating, and recommended index were maintained from the original dataset. Feature engineering was then performed on the review text to generate additional variables for modeling (Figure 2 in References). Since the key feature in the problem is text based and given that it the subjective feelings/opinions of the reviewer, feature engineering was employed to understand and utilize any underlying patterns between target classes. The engineered features of note are the following:
     
- lemma_sent — the tokenized words of each review, cleansed of stop words, punctuation, special characters, and numbers.
- Sentiment scores (polarity, subjectivity, negative, neutral, positive, compound) — textual sentiment scores corresponding to the sentiment, statement bias, negative term score, neutral term score, positive term score, and compound (normalized) sentiment score, respectively.
Minimal data cleansing was needed on the dataset. Only 845 instances had null review text, which were dropped from the dataset since we have sufficient volume of data and since the number of null-valued instances were low.

# Method & Results
For model selection, we set up multiple methods and datasets for experimentation. We chose logistic regression, random forest, and linear support vector classification for their prevalence in NLP problems, simplicity in implementation, and high interpretability. Traditionally, we would have used naïve Bayes, given its use in text classification, however, since our dataset is comprised of both categorical and numerical values, that ruled Naïve Baye’s out from model selection.
For model experimentation and selection, we ran experiments across a combination of:
- 3 datasets
  - text only – a dataset containing only the review text as a feature
  - text and rating – a dataset containing both the review text and review rating
  - all features – a dataset containing the review text, review rating and engineered features
- 3 algorithms – logistic regression, random forest, and linear support vector classification
- 3 types of word-modeling techniques – bag-of-words, n-grams, and term frequency-inverse
document frequency (TF-IDF)

For model performance, recall was used for model evaluation. Since we observed class imbalance during exploratory data analysis, accuracy would not have been an appropriate measure for model evaluation. Given our problem context, recall was chosen as the model evaluation. This allows us to minimize false negatives and maximize the number of instances where the actual label is “not recommended”.
In order to structure our project according to the problem context, the target labels were swapped such that a value of 1 indicated “not recommended” and 0 indicated “recommended”. This allowed us to frame our classification towards the target label of interest: “not recommended” reviews.
Finally, each model experiment was run on the training set using cross-fold validation with the default value for k (5).

After running all experiments on the training set, the model configuration that had the highest weighted average recall and recall on each individual target class was logistic regression using term frequency-inverse document frequency on the text-only dataset. This baseline, candidate model had cross-validated scores of 86% weighted average recall, 87% recall on “recommended” labels, and 82% recall on “not recommended” labels. This effectively allows us, as a business, to ensure that we “capture” 82% of all reviews where the actual label was “not recommended”, further enabling our teams to engage the largest swath of customers possible.
Using the candidate model, final evaluation was then conducted on the test set where it achieved scores of 85% weighted average recall, 86% recall on “recommended” labels, and 82% recall on “not recommended” labels. From these results, we would conclude that we don’t overfit or underfit the dataset and that we have good model generalization.

# Discussion & Future Work
The results achieved were sufficient to warrant using the baseline, version 1 model for the problem at hand. The most enlightening element of the project was that the review rating and engineering features had little effect on the model performance. We posit that because the features of review text (number of tokens, polarity, and subjectivity, etc.) are generally the same across each of the two target classes.

Conducting some basic error analysis, we observe that reviews, even when labeled as “not recommended” contain several positive sentiment tokens (I.e., “love”, “perfect", “great”, etc.). In future modeling, we would take a more stringent approach to text cleansing such that we enable the use of negation in the final review text such that the model can approximate negative sentiment from reviews containing positive sentiment terms.
