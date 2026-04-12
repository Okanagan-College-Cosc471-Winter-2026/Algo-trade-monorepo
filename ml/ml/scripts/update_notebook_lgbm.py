
import json

notebook_path = '/home/cosc-admin/the-project-maverick/ml/notebooks/xgboost_stock_prediction_1.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# 1. Update Install Dependencies Cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "pip install" in "".join(cell['source']):
        source = "".join(cell['source'])
        if "lightgbm" not in source:
             # Append lightgbm to dependency list
            cell['source'] = [line.replace("xgboost", "xgboost lightgbm") if "pip install" in line else line for line in cell['source']]
            # Also add imports
            # We'll rely on the imports cell being separate or part of this.
            # actually, imports are usually in the next cell.
            break

# 2. Update Imports Cell (look for "import pandas")
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "import pandas" in "".join(cell['source']):
        source = "".join(cell['source'])
        if "import lightgbm" not in source:
            new_imports = [
                "import lightgbm as lgb\n",
                "import matplotlib.pyplot as plt\n"
            ]
            cell['source'].extend(new_imports)
        break

# 3. Locate and Replace Training/Tuning Cell
# Finding the cell we previously modified or the one containing search/training logic
target_train_snippet = "Starting hyperparameter tuning..."
train_cell_index = -1

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_code = "".join(cell['source'])
        if target_train_snippet in source_code or "XGBRegressor" in source_code:
            train_cell_index = i
            break

if train_cell_index != -1:
    lgbm_tuning_code = [
        "# LightGBM Model Config & Tuning\n",
        "LGBM_PARAMS_BASE = {\n",
        "    \"objective\": \"regression\",\n",
        "    \"metric\": \"rmse\",\n",
        "    \"boosting_type\": \"gbdt\",\n",
        "    \"random_state\": 42,\n",
        "    \"n_jobs\": -1,\n",
        "    \"learning_rate\": 0.05,\n",
        "    \"n_estimators\": 1000,\n",
        "}\n",
        "\n",
        "lgbm_model = lgb.LGBMRegressor(**LGBM_PARAMS_BASE)\n",
        "\n",
        "param_dist_lgbm = {\n",
        "    \"n_estimators\": [300, 600, 1000, 1500],\n",
        "    \"learning_rate\": [0.01, 0.03, 0.05, 0.1],\n",
        "    \"num_leaves\": [15, 31, 63, 127],\n",
        "    \"max_depth\": [-1, 5, 8, 12],\n",
        "    \"min_child_samples\": [20, 50, 100, 200],\n",
        "    \"subsample\": [0.6, 0.8, 1.0],\n",
        "    \"colsample_bytree\": [0.6, 0.8, 1.0],\n",
        "    \"reg_alpha\": [0, 0.1, 1, 10],\n",
        "    \"reg_lambda\": [0, 0.1, 1, 10],\n",
        "}\n",
        "\n",
        "print(\"Starting LightGBM hyperparameter tuning...\")\n",
        "start_time = time.time()\n",
        "\n",
        "random_search_lgbm = RandomizedSearchCV(\n",
        "    estimator=lgbm_model,\n",
        "    param_distributions=param_dist_lgbm,\n",
        "    n_iter=25,\n",
        "    scoring=\"neg_root_mean_squared_error\",\n",
        "    cv=3,\n",
        "    verbose=1,\n",
        "    random_state=42,\n",
        "    n_jobs=-1\n",
        ")\n",
        "\n",
        "random_search_lgbm.fit(X_train_scaled, y_train)\n",
        "\n",
        "train_duration = time.time() - start_time\n",
        "print(f\"\\nLightGBM tuning completed in {train_duration:.2f} seconds\")\n",
        "print(f\"Best parameters: {random_search_lgbm.best_params_}\")\n",
        "print(f\"Best CV RMSE: {-random_search_lgbm.best_score_:.4f}\")\n",
        "\n",
        "model = random_search_lgbm.best_estimator_\n" # Assign to 'model' to keep compatibility with later cells if they use 'model'
    ]
    nb['cells'][train_cell_index]['source'] = lgbm_tuning_code
    print("Replaced Training/Tuning cell.")

# 4. Update Evaluation Code
# Looking for cell that calculates RMSE/MAPE
target_eval_snippet = "mean_squared_error(y_train"    
eval_cell_index = -1

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and target_eval_snippet in "".join(cell['source']):
        eval_cell_index = i
        break

if eval_cell_index != -1:
    lgbm_eval_code = [
        "# LightGBM Evaluation\n",
        "y_pred_train = model.predict(X_train_scaled)\n",
        "y_pred_test  = model.predict(X_test_scaled)\n",
        "\n",
        "train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))\n",
        "test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))\n",
        "train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100\n",
        "test_mape  = mean_absolute_percentage_error(y_test, y_pred_test) * 100\n",
        "\n",
        "print(\"=\" * 50)\n",
        "print(\"LIGHTGBM EVALUATION METRICS\")\n",
        "print(\"=\" * 50)\n",
        "print(f\"Train RMSE: {train_rmse:.4f}\")\n",
        "print(f\"Test RMSE:  {test_rmse:.4f}\")\n",
        "print(f\"Train MAPE: {train_mape:.2f}%\")\n",
        "print(f\"Test MAPE:  {test_mape:.2f}%\")\n"
    ]
    nb['cells'][eval_cell_index]['source'] = lgbm_eval_code
    print("Replaced Evaluation cell.")

# 5. Update Plotting Code
# Look for matplotlib/plotting, likely next cell or near results
target_plot_snippet = "plt.plot"
plot_cell_index = -1
for i, cell in enumerate(nb['cells']):
     if cell['cell_type'] == 'code' and target_plot_snippet in "".join(cell['source']):
         # Ensure it's the results plot
         if "predicted" in "".join(cell['source']):
             plot_cell_index = i
             break

if plot_cell_index != -1:
    lgbm_plot_code = [
        "results_lgbm = pd.DataFrame({\n",
        "    \"datetime\": test_df[\"date\"].values,\n",
        "    \"actual\": y_test.values,\n",
        "    \"predicted\": y_pred_test,\n",
        "    \"difference\": np.abs(y_test.values - y_pred_test),\n",
        "})\n",
        "\n",
        "plt.figure(figsize=(14, 5))\n",
        "plt.plot(results_lgbm[\"datetime\"], results_lgbm[\"actual\"], label=\"Actual\", alpha=0.7)\n",
        "plt.plot(results_lgbm[\"datetime\"], results_lgbm[\"predicted\"], label=\"Predicted\", alpha=0.7)\n",
        "plt.title(\"LightGBM: Actual vs Predicted Stock Price\")\n",
        "plt.xlabel(\"Date\")\n",
        "plt.ylabel(\"Price ($)\")\n",
        "plt.legend()\n",
        "plt.xticks(rotation=45)\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
    nb['cells'][plot_cell_index]['source'] = lgbm_plot_code
    print("Replaced Plotting cell.")

# 6. Update Save Artifacts Code
target_save_snippet = "joblib.dump(model"
save_cell_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and target_save_snippet in "".join(cell['source']):
        save_cell_index = i
        break

if save_cell_index != -1:
    lgbm_save_code = [
        "OUTPUT_DIR = \"./model_artifacts_lgbm\"\n",
        "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
        "\n",
        "# Save model\n",
        "model.booster_.save_model(f\"{OUTPUT_DIR}/lightgbm_model.txt\")\n",
        "print(f\"Model saved: {OUTPUT_DIR}/lightgbm_model.txt\")\n",
        "\n",
        "# Save scaler (optional but kept to match your pipeline)\n",
        "joblib.dump(scaler, f\"{OUTPUT_DIR}/scaler.joblib\")\n",
        "print(f\"Scaler saved: {OUTPUT_DIR}/scaler.joblib\")\n",
        "\n",
        "# Save feature names\n",
        "feature_cols = X_train.columns.tolist() if hasattr(X_train, 'columns') else [] # Ensure feature_cols is standard\n",
        "joblib.dump(feature_cols, f\"{OUTPUT_DIR}/feature_names.joblib\")\n",
        "print(f\"Feature names saved: {OUTPUT_DIR}/feature_names.joblib\")\n",
        "\n",
        "# Save config\n",
        "config = {\"prediction_horizon\": PREDICTION_HORIZON, \"train_ratio\": TRAIN_RATIO}\n",
        "joblib.dump(config, f\"{OUTPUT_DIR}/config.joblib\")\n",
        "print(f\"Config saved: {OUTPUT_DIR}/config.joblib\")\n",
        "\n",
        "# Save predictions\n",
        "results_lgbm.to_csv(f\"{OUTPUT_DIR}/predictions.csv\", index=False)\n",
        "print(f\"Predictions saved: {OUTPUT_DIR}/predictions.csv\")\n"
    ]
    nb['cells'][save_cell_index]['source'] = lgbm_save_code
    print("Replaced Save cell.")

# 7. Add Inference/Load Cell at the end if not present
# The user provided loading code. We can append it.
lgbm_load_code = [
    "# Load model (for inference)\n",
    "loaded_scaler = joblib.load(f\"{OUTPUT_DIR}/scaler.joblib\")\n",
    "loaded_features = joblib.load(f\"{OUTPUT_DIR}/feature_names.joblib\")\n",
    "\n",
    "loaded_booster = lgb.Booster(model_file=f\"{OUTPUT_DIR}/lightgbm_model.txt\")\n",
    "\n",
    "# sample prediction\n",
    "if hasattr(X_test, 'iloc'):\n",
    "    X_sample = loaded_scaler.transform(X_test.iloc[:5][loaded_features])\n",
    "    sample_pred = loaded_booster.predict(X_sample)\n",
    "    print(\"LightGBM model loaded successfully!\")\n",
    "    print(\"Sample predictions:\", sample_pred)\n"
]

# Check if we should append
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": lgbm_load_code
})
print("Appended Load/Inference cell.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=4)
print("Notebook updated successfully for LightGBM.")
