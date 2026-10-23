# Artifacts Directory

This directory stores reusable preprocessing pipelines, feature metadata, training histories, and decision threshold configurations.

## Artifact Manifest
- `preprocessor_full.joblib`: Fitted `BankPreprocessor` pipeline for all 16 features (post-call).
- `preprocessor_no_duration.joblib`: Fitted `BankPreprocessor` pipeline for 15 features without `duration` (pre-call).
- `feature_names.json`: Names and ordering of one-hot encoded and scaled feature columns.
- `training_histories.json`: Epoch-by-epoch loss, validation loss, and accuracy histories for all ANN architectures.
- `training_metadata.json`: Wall-clock training duration and serialization paths.
- `threshold_optimizations.json`: Validation-derived F1-optimal and business cost-optimal decision thresholds.
- `best_hyperparameters.json`: Optimal hyperparameter configuration discovered via Optuna validation search.
