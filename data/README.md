# Local H&M data

This project uses the [H&M Personalized Fashion Recommendations competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) data. Review and accept the [competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules) before downloading it.

After accepting the rules while signed in to Kaggle, authenticate the Kaggle CLI and download locally:

```bash
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/raw/h-and-m
unzip data/raw/h-and-m/h-and-m-personalized-fashion-recommendations.zip -d data/raw/h-and-m
```

Keep the resulting files local only:

```text
data/
├── raw/h-and-m/{articles.csv,customers.csv,transactions_train.csv,images/}
└── processed/{hm_customer_cohort.csv,hm_customer_cohort.summary.json}
```

Neither source rows nor processed rows may be committed to this repository or redistributed. Product images and Kaggle credentials are also local-only.
