# Data

The H&M data is not stored in this repo (Kaggle competition rules). To rebuild everything:

1. Accept the rules of the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) competition on Kaggle.
2. Put these three files in `data/raw/` (CSV from the competition converted to parquet, or the ready-made parquet copy at [derrickmwiti/hm-personalized-fashion-recommendations](https://www.kaggle.com/datasets/derrickmwiti/hm-personalized-fashion-recommendations)):
   - `transactions_train.parquet` (31,788,324 rows)
   - `customers.parquet` (1,371,980 rows)
   - `articles.parquet` (105,542 rows)
3. Run `make sample`. It writes to `data/processed/`:
   - `transactions_sample.parquet`: 2020-06-24 to 2020-09-22, customers whose hashed id ends in `00` to `33` (52 of 256 buckets, about 20%)
   - `customers_sample.parquet`: the same customers
   - `articles.parquet`: all articles
   - `daily_full.parquet`: daily counts over the full two years, for the overview chart

`make tables` then writes the per-week candidate and feature tables to `data/processed/tables/`, and `make serve` writes the API artifacts to `data/processed/serving/`.

Column definitions: [`docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md).
