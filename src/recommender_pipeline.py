
# Load the movies dataset
movies_df = spark.read.option("header", "true") \
                      .option("inferSchema", "true") \
                      .csv("/FileStore/tables/movies.csv")
movies_df.cache()  # Cache for faster operations

# Inspect the schema and first few rows
movies_df.printSchema()
display(movies_df)

# Calculate average rating and count of ratings per movie
from pyspark.sql.functions import avg, count

movie_stats_df = movies_df.groupBy("movieId").agg(
    avg("rating").alias("avg_rating"),
    count("rating").alias("num_ratings")
)

# Identify the top 10 movies by average rating
top10_movies = movie_stats_df.orderBy("avg_rating", ascending=False).limit(10)

display(top10_movies)
print("Above are top 10 movies with the highest average ratings")

# Identify the top 10 users by the number of ratings they have provided
top10_users = movies_df.groupBy("userId").agg(
    count("rating").alias("rating_count")
).orderBy("rating_count", ascending=False).limit(10)

display(top10_users)
print("Above are top 10 users with the highest number of ratings")


import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import count

# 1. Count how many ratings each user provides
user_rating_counts = movies_df.groupBy("userId").agg(
    count("rating").alias("num_ratings")
)

# 2. Convert to Pandas for easier plotting
user_rating_counts_pd = user_rating_counts.toPandas()

# 3. (Optional) Sort the DataFrame by userId or by num_ratings
#    Here we sort by userId for a cleaner left-to-right display
user_rating_counts_pd = user_rating_counts_pd.sort_values(by="userId")

# 4. Plot a bar chart with userId on the x-axis and the number of ratings on the y-axis
plt.figure(figsize=(10, 6))
sns.barplot(data=user_rating_counts_pd, x="userId", y="num_ratings", palette="Blues_d")

# 5. Configure labels and aesthetics
plt.title("Number of Ratings per User")
plt.xlabel("User ID")
plt.ylabel("Number of Ratings")
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.tight_layout()
plt.show()


# 1. Overall Distribution of Ratings
# Convert the ratings column from Spark DataFrame to a Pandas DataFrame
ratings_pdf = movies_df.select("rating").toPandas()

plt.figure(figsize=(8,6))
sns.histplot(ratings_pdf['rating'], bins=10, kde=True, edgecolor="black")
plt.title("Overall Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()





# COMMAND ----------

# MAGIC %md
# MAGIC From the table 1, we can see that there are 3 headers in this movie dataset. They are movieId, rating and userId, which are integer. Table 2 shows top 10 movies with highest rating and table 3 shows top 10 user with most frequent raing. From the visualization of number os ratings per user, we can see that all users rated over 40 times. And there are 11 users rated over 50 times. From the visualization of overall distribution of ratings, we can see that the number of rate 1.0 is most rated by users while rate 5.0 is the least. 

# COMMAND ----------

# MAGIC %md
# MAGIC For the potential implications of marketing strategies, because the users all were actived and engaged, we can offer more services to the users such as group discussion about movie or other paid service. In addtion to this, we can see that rate 1.0 are more than rate 5.0. We can ask the users why the movie is rate 1.0 to help develop better movie service. And we can also make personalized targeting. We can learn what categories are highly rated by each user, and then recommend the same or similar category of movies to them.

# COMMAND ----------

#partB Q2
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# Define a list of split ratios (training portion)
split_ratios = [0.6, 0.7, 0.75, 0.8]
results = {}

for ratio in split_ratios:
    # Split the dataset into training and test sets based on the ratio
    train, test = movies_df.randomSplit([ratio, 1 - ratio], seed=42)
    
    # Initialize the ALS model with chosen hyperparameters
    als = ALS(
        userCol="userId", 
        itemCol="movieId", 
        ratingCol="rating", 
        coldStartStrategy="drop",  # drop predictions for unknown users/items
        nonnegative=True,
        maxIter=10,
        regParam=0.1,
        rank=10
    )
    
    # Train the model on the training split
    model = als.fit(train)
    
    # Generate predictions on the test set
    predictions = model.transform(test)
    
    # Evaluate using RMSE
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Training/Test Ratio {ratio*100:.0f}/{(1-ratio)*100:.0f} RMSE: {rmse:.4f}")
    
    results[ratio] = rmse


# COMMAND ----------

# MAGIC %md
# MAGIC From above table, we can see that as the training set increases, the RMSE on test set decreases. It means the model is learning more effectively with additional training samples. Among the above tested ratios, because the 80/20 split has the lowest RMSE (0.9275), i think it is the most effective split configuration for this dataset.

# COMMAND ----------

#partB Q3
# use an 80/20 split
train, test = movies_df.randomSplit([0.8, 0.2], seed=42)
als = ALS(
    userCol="userId", 
    itemCol="movieId", 
    ratingCol="rating", 
    coldStartStrategy="drop", 
    nonnegative=True,
    maxIter=10,
    regParam=0.1,
    rank=10
)
model = als.fit(train)
predictions = model.transform(test)

# Evaluate RMSE
evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator_rmse.evaluate(predictions)

# Evaluate MAE
evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
mae = evaluator_mae.evaluate(predictions)

# Compute MSE (MSE is RMSE squared)
mse = rmse ** 2

print("RMSE:", rmse)
print("MAE:", mae)
print("MSE:", mse)

import pyspark.sql.functions as F

# Define a threshold to binarize the ratings (ratings >= 3.0 are considered 'relevant')
threshold = 3.0

# Create binary labels for actual and predicted ratings
predictions_with_labels = predictions.withColumn(
    "actual_label", F.when(F.col("rating") >= threshold, 1).otherwise(0)
).withColumn(
    "predicted_label", F.when(F.col("prediction") >= threshold, 1).otherwise(0)
)

# Calculate counts for True Positives, False Positives, and False Negatives
metrics = predictions_with_labels.agg(
    F.sum(F.when((F.col("predicted_label") == 1) & (F.col("actual_label") == 1), 1).otherwise(0)).alias("TP"),
    F.sum(F.when((F.col("predicted_label") == 1) & (F.col("actual_label") == 0), 1).otherwise(0)).alias("FP"),
    F.sum(F.when((F.col("predicted_label") == 0) & (F.col("actual_label") == 1), 1).otherwise(0)).alias("FN")
).collect()[0]

TP = metrics["TP"]
FP = metrics["FP"]
FN = metrics["FN"]

# Compute Precision, Recall, and F1 Score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)



# COMMAND ----------

# MAGIC %md
# MAGIC The RMSE is 0.9275. A lower RMSE indicates the model’s predicted ratings are relatively close to actual ratings on average. The MAE is 0.6126. It is more robust to outliers than RMSE because large errors are not squared. The MSE is 0.8603. MSE is the average of the squared errors and is the square of the RMSE. The precision is 0.75, which means all items the model predicted as “liked” (≥ 3.0), 75% were truly liked by the user. The recall is 0.1452, which means in the items that were truly liked by the user, the model only identifies about 14.5% of them. It means the users may miss many movies. The F1 score is 0.2432. F1 score is harmonic mean of Precision and Recall. For this recommendation system, it relies solely on user ratings. RMSE, MAE, and MSE are regression metrics measure how close predicted ratings are to the actual ones, but they don’t show whether users truly “like” a movie. Classification metrics: Precision, Recall, and F1, which are better for assessing how well the recommended list captures user preferences, especially under a binary rule like “rating ≥ 3.0.” In imbalanced datasets, Recall often ends up low, meaning the system may miss many items users would actually like. If broaden recommendations to boost Recall, Precision tends to drop. Ultimately, which metrics we prioritize depends on our goals: if we want to accurately predict the exact rating, focus on lowering RMSE or MAE; if we want users to see as many liked items as possible, raise Recall but also keep an eye on Precision to balance coverage and accuracy.

# COMMAND ----------

#partB Q4
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Split dataset (using an 80/20 train-test split)
train, test = movies_df.randomSplit([0.8, 0.2], seed=42)

# Initialize ALS model (without fixed hyperparameters)
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", 
          coldStartStrategy="drop", nonnegative=True)

# Build the parameter grid for tuning
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [8, 10, 12]) \
    .addGrid(als.regParam, [0.05, 0.1, 0.15]) \
    .addGrid(als.maxIter, [10, 15]) \
    .build()

# Define the evaluator using RMSE metric
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Set up 3-fold cross-validation with the grid
cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

# Fit the cross-validator on the training data
cvModel = cv.fit(train)

# Retrieve the best model and evaluate it on the test set
bestModel = cvModel.bestModel
best_rmse = evaluator.evaluate(bestModel.transform(test))
print("Best RMSE from Cross-Validation: {:.4f}".format(best_rmse))


# Extract hyperparameter combinations and their average RMSE scores
avgMetrics = cvModel.avgMetrics
param_list = []
for params, metric in zip(paramGrid, avgMetrics):
    param_dict = {}
    for param, value in params.items():
        param_dict[param.name] = value
    param_dict['rmse'] = metric
    param_list.append(param_dict)

results_df = pd.DataFrame(param_list)
print(results_df)


# Visualize the impact of different hyperparameters on RMSE
# Here we create a heatmap for each fixed maxIter value, plotting RMSE vs. rank and regParam
for max_iter in sorted(results_df['maxIter'].unique()):
    subset = results_df[results_df['maxIter'] == max_iter]
    pivot_table = subset.pivot(index='rank', columns='regParam', values='rmse')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title(f"RMSE Heatmap (maxIter = {max_iter})")
    plt.xlabel("regParam")
    plt.ylabel("rank")
    plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC  From the first heatmap (maxIter=15), we can see that when rank=8 and regParam=0.05, the RMSE is as high as 1.2474. As regParam increases to 0.15, the RMSE drops to 1.1409, indicating that appropriately increasing regularization helps alleviate overfitting and improves accuracy. When rank=10 or rank=12, the RMSE ranges roughly between 1.18 and 1.14, and setting regParam to 0.1 or 0.15 yields a relatively low RMSE (around 1.15). Comparing this with the second heatmap (maxIter=10), under fewer iterations, if rank=8 and regParam=0.05, the RMSE rises to 1.2712, which is worse. However, with rank=12 and regParam=0.1 or 0.15, the RMSE can be kept near 1.15, suggesting that increasing rank along with moderate regularization can maintain decent accuracy under fewer iterations—though it’s still slightly higher than the best RMSE (about 1.14) achieved when maxIter=15. Overall, maxIter=15 allows the model to converge further, especially with a higher rank and moderate regParam, achieving a lower RMSE but at a higher training cost. If you need to balance training time and accuracy, you might consider reducing the number of iterations or choosing a moderate rank and moderate regularization to find an optimal trade-off between performance and efficiency.

# COMMAND ----------

#partB Q5
# 1. create a usersID's DataFrame
target_users = spark.createDataFrame([(11,), (21,)], ["userId"])
# 2. use the bestModel on step4 to recommend 5 movies
user_recommendations = bestModel.recommendForUserSubset(target_users, 5)
display(user_recommendations)
# 3. baseline: just pick the top 5 movies with highest average rating.
from pyspark.sql.functions import count

popular_movies = top10_movies.limit(5)
display(popular_movies)


# COMMAND ----------

# MAGIC %md
# MAGIC Collaborative filtering can generate personalized recommendations by leveraging user ratings to find patterns and similarities between users. It works well in identifying niche preferences but faces challenges in data-sparse situations or new users and items. Enhancing collaborative filtering with additional features like movie genres, temporal dynamics, or contextual data can improve personalization, while incorporating better ranking metrics can refine future iterations for even more effective recommendations. While Collaborative filtering provides more personalized suggestions compared to the methods we analysis before such as just recommending the most popular movies, it may struggle in sparse datasets or imbalanced ratings. 

# COMMAND ----------


