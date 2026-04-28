# Mini Project 2 Report Template

## 1) Dataset Description and Visualization
- Describe UA-DETRAC splits used.
- Vehicle classes: car, van, bus, others.
- Include qualitative examples with GT bounding boxes.

## 2) Data Preprocessing
- XML parsing into detection records.
- Label encoding.
- Train/validation split policy.
- Any augmentations / resizing / normalization.

## 3) Training Process and Hyperparameter Tuning
- Model architecture and why chosen.
- Losses and optimizer.
- Hyperparameters table (lr, batch size, epochs, confidence threshold, etc.).
- Include plots:
  - Training and Validation loss curves.
  - Validation F1 score per epoch.

## 4) Results and Evaluation
- Metrics on test set:
  - Precision
  - Recall
  - F1-Score
  - mAP
- Qualitative visualization on unseen/test video.
- Traffic statistics from inference:
  - Maximum load of road (max vehicles in one frame).
  - Maximum traffic flow in selected time window.

## 5) Conclusion
- Main findings.
- Limitations and failure cases.
- Improvements (tracking-based flow counting, stronger detector, better augmentation).
