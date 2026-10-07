# Model card: H&M next-week product recommender

## Model

- **Type**: two-stage recommender. Stage 1: seven candidate generators (best sellers, age-band best sellers, buy again, item-to-item CF, ALS, PCA content-based, other colors). Stage 2: LightGBM binary classifier that scores P(customer buys item next week)
- **Files**: `models/lgbm_classifier.joblib` (and `.txt` booster), feature list, categories and tuned settings in `models/model_meta.json`
- **Settings**: 127 leaves, learning rate 0.05, min child samples 200, colsample 0.7, subsample 0.6, L2 5, 137 rounds (early stopping on validation AUC)
- **Version**: 1.0.0, seed 42, config `configs/config.yaml`

## Intended use

Ranking 12 products per customer for "You may also like" rows, app home and campaign emails, refreshed weekly by a batch job. Not meant for pricing, credit or any decision about a person beyond which products to show.

## Data

H&M Personalized Fashion Recommendations (Kaggle, 2022). Last 13 weeks (24 Jun to 22 Sep 2020), about 20% of customers (fixed hash rule). Training labels from weeks 2 to 4, tuning on week 1, test on week 0 (16 to 22 Sep 2020). Features only use the 8 weeks before each target week.

## Performance (test week, 14,023 customers)

| Metric | LightGBM ranker | Best-seller baseline |
|---|---|---|
| MAP@12 | 0.0309 | 0.0094 |
| Recall@12 | 0.0663 | 0.0284 |
| Hit rate | 13.1% | 7.1% |
| AUC (candidates) | 0.758 | |
| Customers with history: MAP@12 | 0.0448 | 0.0101 |
| Customers without history: MAP@12 | 0.0086 | 0.0083 |

Candidate recall is 11.2%, which caps what the ranker can find.

## Fairness

| Age band | Customers | MAP@12 | Hit rate | Disparate impact |
|---|---|---|---|---|
| 16-24 | 3,824 | 0.0283 | 12.0% | 0.81 |
| 25-34 | 4,242 | 0.0283 | 13.0% | 0.88 |
| 35-44 | 1,467 | 0.0317 | 13.1% | 0.88 |
| 45-54 | 2,607 | 0.0335 | 13.9% | 0.93 |
| 55+ | 1,827 | 0.0382 | 14.8% | 1.00 |

- Equalized odds (top 12 as positive): TPR gap 0.089, FPR gap 0.012. Demographic parity difference 0.012
- Item side: 95.3% of slots go to the top 20% sellers, while 34.1% of purchases are long tail
- Removing the age features costs 0.8% MAP@12 and raises the lowest disparate impact from 0.81 to 0.86. Recommended for production
- Gender, race and income are not in the data and were not inferred. Postal code was dropped

## Limitations

20% customer sample; 38% of test buyers have no recent history; prices are scaled and there is no margin data; offline metrics need an A/B test to confirm online uplift.

## Monitoring

Weekly MAP@12 and hit rate overall and by age band (alert below 0.8 disparate impact), long-tail share of slots, feature drift on the top SHAP features, data freshness. See `docs/DEPLOYMENT.md`.
