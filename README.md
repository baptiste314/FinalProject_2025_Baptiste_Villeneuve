# Final Project Report: KuaiRec Short Video Recommender System

**By:** Baptiste Villeneuve

**ID:** baptiste.villeneuve

## 0. Code Usage and Submission
* I used `python3.8` for this project, and I installed the libraries in `requirements.txt`
* This `README.md` file is the report
* The submission file is in `outputs/submission.csv`
* The neural network model is already trained, the file is in `models/nn_recommender_model.h5`. At some point the code checks if this file exists, if not it recreates the NN and train it (takes a lot of time).
* The code checks if `data_final_project/KuaiRec (or Kuairec (2.0))/data` exists. If not, it download it from the google drive, unzip it and creates `data_final_project/KuaiRec (or Kuairec (2.0))/data`


## 1. Introduction

**Problem explanation:** The goal of this project is to develop a personalized recommender system for short videos using the KuaiRec dataset. The main task was to predict user engagement, the `watch_ratio` (play duration / video duration), for user-video pairs.

**Dataset:** The KuaiRec dataset provided user-video interactions (views, durations), video metadata (captions, categories, tags), and user features (demographics, activity). A 5-million interaction sample from `big_matrix.csv` was used for training/validation, and `small_matrix.csv` for final test predictions.

**Objective:** To build and evaluate a model that predicts `watch_ratio` and therefore provides relevant video recommendations.

## 2. Methodology

Here are the steps that I did to complete the project :

### 2.1. Data Exploration & Preprocessing
*   **Data Analysis:** The initial analysis shows :
    - A large interaction matrix 
    - Most user-video interactions were missing (83.28% empty).
    - A few users and videos had many interactions, while most had few.
    - The watch_ratio (how much of a video was watched) varied widely
    - Patterns based on video categories and user activity levels

*   **Preprocessing Steps:**
    *   **Data Merging:** I merged interaction data (`big_matrix.csv`) with video metadata (`kuairec_caption_category.csv`) and user features (`user_features.csv`) to create a more global dataset.
    *   **Type Handling:** I made sure data types are consistent to merge keys (`user_id`, `video_id`).
    *   **Missing Value Imputation:**
        *   Missing  `watch_ratio` values: replaced with the median value.
        *   Missing categorical features: filled with 'Unknown'.
        *   Missing numerical user features (one-hot): Filled with 0.
    *   **Data Cleaning:** Here I removed duplicate interactions and interactions with invalid video durations (<=0), and I recomputed `watch_ratio` where it's necessary.
    *   **Memory Optimization:** Because of the huge amount of data, I had to downcast numerical data (for example convert from float64 to float32).

### 2.2. Feature Engineering
I created features to reflect the different factors that affect user engagement.
*   **Temporal Features:** `hour`, `day_of_week`, `is_weekend` obtained from interaction timestamps.
*   **User Aggregates:** `user_interaction_count`, `user_avg_watch_ratio`
*   **Video Aggregates:** `video_interaction_count`, `video_avg_watch_ratio`, `video_distinct_users_watched`
*   **Categorical Encoding:** One-Hot Encoding for features `hour`, `day_of_week`, `first_level_category_name`, `user_active_degree`.
*   **Text Encoding:** TF-IDF (max 50 features for `caption`, 25 for `topic_tag`) for video textual metadata.
*   **Numerical Scaling:** All numeric features were standardized using `StandardScaler` for the model.
*   **Final Feature Set:** All features were combined into a single sparse matrix, which gives us 157 features for model input.
*   **Goal:** Now that we have this wide range of informations, the model will be able to predict `watch_time`.

### 2.3. Model Development
*   **Baseline Model:** For the baseline model, I created an "Item Average" model that makes predictions based on average value. It predicts a video's average `watch_ratio` from the training set (or global average for new videos not seen in training set).

    *   Validation RMSE: 1.6921
    *   MAE: 0.5428.
*   **Chosen Model: Neural Network (NN):**
    *   **Experiments** I tried to do a LightGBM model and a XGBoost model but I was getting slightly better results with a Neural Network, that's why I'm going with it.
    *   **Architecture:** A feedforward neural network with a sparse input layer (157 features)!
        *   Dense layer (128 units, ReLU, L2 )
            *   Dropout (0.3)
        *   Dense layer (64 units, ReLU, L2)
            *   Dropout (0.3)
        *   Output Dense layer (1 unit, regression).
    *   **Loss Function:** Mean Squared Error (MSE).
    *   **Optimizer:** Adam (learning rate 0.001).
    *   **Training:** Trained on the 80% training split of the 5M sample, using an 20% validation split for early stopping (early stopping based on `val_rmse`, if no improvment after 10 epoch, and early stopping restores best weights). Batch size was 256. The best model was achieved at epoch 16.
    *   **Why NN:** Neural networks are suitable for this task. As I said earlier, I tried other models like LightGBM (Val RMSE: 1.6895) and XGBoost (Val RMSE: 1.6839) which showed comparable regression performance, but I choosed the NN because its RMSE was slightly btetter

## 3. Results and Evaluation

The final Neural Network model was evaluated on the validation set:

*   **Regression Metrics:**
    *   **RMSE:** 1.6691
    *   **MAE:** 0.5417
*   **Ranking Metrics (Top-10, `watch_ratio >= 1.0`):**
    *   **Precision@10:** 0.7097
    *   **Recall@10:** 0.1142
    *   **NDCG@10:** 0.4739

**Interpretation:**
*   The NN model (RMSE: 1.6691) showed a modest improvement in predictive accuracy for `watch_ratio` compared to the Item Average baseline (RMSE: 1.6921). The MAE was similar.
*   A **Precision@10 of ~0.71** is a good result, it means that when 10 videos are recommended, users are likely to find approximately 7 of them engaging.
*   The **Recall@10 of ~0.11** means that the top 10 recommendations cover a smaller portion of all potentially relevant videos for a user. This may be because of the large amount of videos and users.
*   An **NDCG@10 of ~0.47** means that the model ability to put the most relevent item at the top of the list is ok, but not very good.

## 4. Conclusion

This project successfully developed a Neural Network-based recommender system for the KuaiRec dataset. We created a wide range of engineered features related to how users behave, what's in the videos, and how users interact with them, the model was able to predict how much of a video a user would watch (watch_ratio) and suggest videos that are relevant to them.

The final NN model outperformed a simple item average baseline in terms of RMSE and showed great precision in its top-10
recommendations. 
The overall predictive accuracy (RMSE) and recall can still be improved.
The challenges were to manage the large dataset, to perform the feature engineering while avoiding leaks.

Future work to improve : 
* Try differents model architectures
* Handle cold start
* Tune hyperparameters
* More features/better ones

