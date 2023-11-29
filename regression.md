Linear regression with inference.

 1. EDA.
   Firstly, the data set was prepared for working with it: i.e. the duplicated from the df_train were deleted, all Nan elements were replaced by the median of the corresponding column of the     
   df_train (both for df_train and df_test), the column "torque" was deleted, units of measurements were deleted from the columns "engine", "mileage" and "max_power". Then, the data was 
   visualized (i.e. the plots and heatmap were built for to understand the correlations between the features (both for df_train and df_test).
 2. Then linear regression model was used for the data set df_train only with numerical features (all features except "fuel", "seller_type", "transmission", "owner", "name"). On train R2 score 
    was 0.5922591702157316, MSE was 116874153930.02855. On test R2 score was 0.5941419794788428, MSE was 233298779730.45486.
 3. The data was also normalized. R2_score on train was 0.5922591702157303 and 0.594141979478852 on test. The most informative feature was "max_power". Then Lasso Regression was used (where the      best coefficients were found using GridSearch). Best value for alpha is 26609. For this alpha 3 of 6 coefficients are zero.
 5. The same was done for ElasticNet Regression. Best value for alpha and l1_ratio : {'alpha': 26610, 'l1_ratio': 0.9999999999999944} Best score for cost function (r2 score): 0.5750274764701039
 6. After that the linear regression was used for all features inclidung categorical. For train: r2_score for y_train: 0.6565982103937898, MSE for y_train: 98432118361.84863. For test r2_score 
   for y_test: 0.64462236110295, MSE for y_test: 204281214873.39188
 7. After that severals attempts were taken to make the metrics better. The best results gave the following models (each for Rigde Regression):
    --model3-l2: a new column ("square of year") was added and the column "year" was removed (since it gave better results on test).
    --model4-l2: a new colun ("one over km_driven") was added and the column "km_driven" was removed. This model gave slightly worse results.
    --model5-l2: a new column ("max_power/engine") was added. This model gave slighly better results than the previous ones (but not significantly)
    --model6-l2: Now we tried to create a new column with 1 if owner is first owner or it is a test drive car, and 0 otherwise. We also creat a new column if seller_type is dealer and 0 
    otherwise. We also use the square of year instead of year. This model gave again slighly better results that the previous ones: r2_score for y_test: 0.6455001683347408
    MSE for y_test: 203776626210.21262
  8. The function that counts customs metric was written (counts the proportions of prediction that differes from actual values not more than by 10%).

     
