# Titanic Passenger Insights

An interactive Streamlit dashboard for exploring survival patterns in the Titanic
dataset, with the data-preparation steps from the activity brief: missing-value
handling, Min-Max normalization, age discretization, and dummy coding.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

| Tab | Content |
|---|---|
| Overview | 4 KPI cards, survival counts, survival rate by class |
| Explore | Survival rate by class and sex, age histogram, fare by class, finding |
| Preprocessing | Missing values, Min-Max normalization, age discretization, dummy coding |
| Data Explorer | Raw / Cleaned / Transformed views, column picker, load-and-inspect |

## Filters

Sidebar selectboxes for passenger class, sex, and age group. Every KPI and chart
reads from the filtered frame.

## Notes

- `age` is filled with the median (28.0) and `embarked` with the most frequent
  category (S). `deck` is excluded: 688 of 891 values are missing.
- `age_group`: 0-12 Child, 13-18 Teen, 19-35 Young Adult, 36-60 Adult, 61+ Older Adult.
- Dummy variables are created for `sex` and `embarked` only. `survived` is the
  target and is not encoded; dummy columns are not normalized.
- Rates describe this dataset only. The charts show associations, not causation.
