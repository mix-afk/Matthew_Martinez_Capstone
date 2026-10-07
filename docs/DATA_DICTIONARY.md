# Data dictionary

Source: H&M Personalized Fashion Recommendations (Kaggle competition data, parquet copy).

## transactions

| Variable | Type | Unit | Allowed values / range | Description |
|---|---|---|---|---|
| t_dat | date (string in raw file) | day | 2018-09-20 to 2020-09-22 (sample: 2020-06-24 to 2020-09-22) | Purchase date. One row per unit sold |
| customer_id | string (64-char hex hash) |  | Matches customers.customer_id | Anonymized customer key |
| article_id | int64 |  | Matches articles.article_id | Product variant key (product + color) |
| price | float64 | scaled currency | 0.00002 to 0.5915 (scaled by H&M, not real money) | Unit price paid |
| sales_channel_id | int64 |  | 1, 2 (2 is widely read as online, 1 as store) | Sales channel |

## customers

| Variable | Type | Unit | Allowed values / range | Description |
|---|---|---|---|---|
| customer_id | string (64-char hex hash) |  | Unique per row | Anonymized customer key |
| FN | float64 (flag) |  | 1.0 or missing (missing = 0) | Customer gets fashion news |
| Active | float64 (flag) |  | 1.0 or missing (missing = 0) | Customer is active for communication |
| club_member_status | string |  | ACTIVE, PRE-CREATE, LEFT CLUB, missing | Loyalty club status |
| fashion_news_frequency | string |  | NONE, None, Regularly, Monthly, missing (None and missing merged into NONE) | Newsletter frequency |
| age | float64 | years | 16 to 99, about 1.2% missing | Customer age. Used for the fairness audit |
| postal_code | string (hash) |  | Hashed, 352,899 distinct values | Postal code. Dropped (too granular, possible location proxy) |

## articles

| Variable | Type | Unit | Allowed values / range | Description |
|---|---|---|---|---|
| article_id | int64 |  | 105,542 unique | Product variant key |
| product_code | int64 |  | 47,224 unique | Parent product (all colors of one item share it) |
| prod_name | string |  | Free text | Product name |
| product_type_name | string |  | 131 values, e.g. Trousers, Sweater, Dress | Product type (paired code: product_type_no) |
| product_group_name | string |  | 19 values, e.g. Garment Upper body, Accessories | Product group |
| graphical_appearance_name | string |  | 30 values, e.g. Solid, Stripe, All over pattern | Print or pattern |
| colour_group_name | string |  | 50 values, e.g. Black, Light Beige | Detailed color (paired code: colour_group_code) |
| perceived_colour_value_name | string |  | 8 values, e.g. Dark, Light, Dusty Light | Color tone |
| perceived_colour_master_name | string |  | 20 values, e.g. Black, Blue, Beige | Main color family |
| department_name | string |  | 250 values | Buying department |
| index_name | string |  | 10 values, e.g. Ladieswear, Divided, Menswear | Store index |
| index_group_name | string |  | 5 values: Ladieswear, Baby/Children, Divided, Menswear, Sport | Top-level index group |
| section_name | string |  | 56 values | Section within index |
| garment_group_name | string |  | 21 values, e.g. Jersey Basic, Knitwear | Garment group |
| detail_desc | string |  | Free text, 416 missing | Product description |
