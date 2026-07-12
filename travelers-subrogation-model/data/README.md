# Data

The 2025 Travelers University Modeling Competition ("TriGuard") data files are
not redistributed in this repository. Place the competition files here:

```
data/Training_TriGuard.csv
data/Testing_TriGuard.csv
```

Both files are excluded via `.gitignore` so they never get committed.

Expected columns (raw, before feature engineering) include: `claim_number`,
`subrogation` (target, training only), `claim_date`, `claim_day_of_week`,
`year_of_born`, `age_of_DL`, `liab_prct`, `vehicle_made_year`, `vehicle_price`,
`vehicle_weight`, `vehicle_mileage`, `annual_income`, `claim_est_payout`,
`safety_rating`, `past_num_of_claims`, plus categorical fields such as
`gender`, `living_status`, `accident_site`, `accident_type`,
`witness_present_ind`, `policy_report_filed_ind`, `vehicle_category`,
`vehicle_color`, `in_network_bodyshop`, `channel`, `zip_code`,
`high_education_ind`, `address_change_ind`, `email_or_tel_available`.
