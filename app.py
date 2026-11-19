# -*- coding: utf-8 -*-
"""Titanic Passenger Insights — Streamlit dashboard.

Follows the activity brief: four tabs (Overview, Explore, Preprocessing,
Data Explorer), sidebar filters, and the four data-preparation demos.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Titanic Passenger Insights",
    page_icon="🚢",
    layout="wide",
)

# --- Palette: navy with teal accents ---------------------------------------
NAVY = "#1f3a5f"
TEAL = "#2a9d8f"
GREY = "#b8c4ce"
SEX_COLORS = {"female": TEAL, "male": NAVY}
CLASS_COLORS = {1: NAVY, 2: TEAL, 3: GREY}
AGE_ORDER = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
FIELDS = ["survived", "pclass", "sex", "age", "fare", "embarked"]


# ============================================================================
# Data
# ============================================================================
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")


def prepare(raw):
    """Clean the raw frame and add the transformed features."""
    clean = raw.copy()

    # Missing values: median for age, most frequent category for embarked.
    age_median = raw["age"].median()
    embarked_mode = raw["embarked"].mode()[0]
    clean["age"] = clean["age"].fillna(age_median)
    clean["embarked"] = clean["embarked"].fillna(embarked_mode)

    # Min-Max normalization, originals kept.
    clean["age_norm"] = (clean["age"] - clean["age"].min()) / (
        clean["age"].max() - clean["age"].min()
    )
    clean["fare_norm"] = (clean["fare"] - clean["fare"].min()) / (
        clean["fare"].max() - clean["fare"].min()
    )

    # Discretization. right=False so the ranges match the brief exactly.
    clean["age_group"] = pd.cut(
        clean["age"],
        bins=[0, 13, 19, 36, 61, float("inf")],
        labels=AGE_ORDER,
        right=False,
    )
    return clean, age_median, embarked_mode


def bar_chart(series, title, ylabel, colors, value_fmt="{:.2f}"):
    """One bar per category, labeled axes, consistent colors."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar(
        [str(i) for i in series.index], series.values, color=colors, width=0.6
    )
    ax.set_title(title, color=NAVY, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel, color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    for bar, value in zip(bars, series.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            value_fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color=NAVY,
        )
    fig.tight_layout()
    return fig


raw = load_data()
clean, age_median, embarked_mode = prepare(raw)
dummies = pd.get_dummies(clean[["sex", "embarked"]], columns=["sex", "embarked"])

# ============================================================================
# Sidebar filters
# ============================================================================
with st.sidebar:
    st.header("Filters")
    pclass_sel = st.selectbox("Passenger class", ["All", 1, 2, 3])
    sex_sel = st.selectbox("Sex", ["All", "female", "male"])
    age_sel = st.selectbox("Age group", ["All"] + AGE_ORDER)

mask = pd.Series(True, index=clean.index)
if pclass_sel != "All":
    mask &= clean["pclass"] == pclass_sel
if sex_sel != "All":
    mask &= clean["sex"] == sex_sel
if age_sel != "All":
    mask &= clean["age_group"] == age_sel

filtered = clean[mask]

# ============================================================================
# Header
# ============================================================================
st.title("Titanic Passenger Insights")
st.caption("Explore survival patterns across passenger groups")

if filtered.empty:
    st.warning("No passengers match the current filters. Widen your selection.")
    st.stop()

tab_overview, tab_explore, tab_prep, tab_data = st.tabs(
    ["Overview", "Explore", "Preprocessing", "Data Explorer"]
)

# ============================================================================
# Tab 1 — Overview
# ============================================================================
with tab_overview:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Passengers in selection", f"{len(filtered):,}")
    k2.metric("Survivors", f"{int(filtered['survived'].sum()):,}")
    k3.metric("Survival rate", f"{filtered['survived'].mean():.1%}")
    k4.metric("Average fare", f"{filtered['fare'].mean():.2f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        counts = (
            filtered["survived"]
            .map({0: "Died", 1: "Survived"})
            .value_counts()
            .reindex(["Died", "Survived"])
            .fillna(0)
        )
        st.pyplot(
            bar_chart(
                counts,
                "Survival counts",
                "Passengers",
                [GREY, TEAL],
                value_fmt="{:.0f}",
            )
        )
    with c2:
        by_class = filtered.groupby("pclass", observed=True)["survived"].mean()
        st.pyplot(
            bar_chart(
                by_class,
                "Survival rate by passenger class",
                "Survival rate",
                [CLASS_COLORS[c] for c in by_class.index],
            )
        )

    st.caption("Rates describe this dataset only.")

# ============================================================================
# Tab 2 — Explore
# ============================================================================
with tab_explore:
    st.subheader("Compare passenger groups")

    c1, c2 = st.columns(2)
    with c1:
        by_class = filtered.groupby("pclass", observed=True)["survived"].mean()
        st.pyplot(
            bar_chart(
                by_class,
                "Survival rate by passenger class",
                "Survival rate",
                [CLASS_COLORS[c] for c in by_class.index],
            )
        )
    with c2:
        by_sex = filtered.groupby("sex", observed=True)["survived"].mean()
        st.pyplot(
            bar_chart(
                by_sex,
                "Survival rate by sex",
                "Survival rate",
                [SEX_COLORS[s] for s in by_sex.index],
            )
        )

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(filtered["age"], bins=20, color=TEAL, edgecolor="white")
        ax.set_title(
            "Age distribution", color=NAVY, fontsize=12, fontweight="bold", loc="left"
        )
        ax.set_xlabel("Age", color=NAVY)
        ax.set_ylabel("Passengers", color=NAVY)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=NAVY)
        fig.tight_layout()
        st.pyplot(fig)
    with c4:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        data = [filtered.loc[filtered["pclass"] == c, "fare"] for c in [1, 2, 3]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55)
        for patch, c in zip(bp["boxes"], [1, 2, 3]):
            patch.set_facecolor(CLASS_COLORS[c])
        ax.set_xticks([1, 2, 3], ["1st", "2nd", "3rd"])
        ax.set_title(
            "Fare by passenger class",
            color=NAVY,
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax.set_xlabel("Passenger class", color=NAVY)
        ax.set_ylabel("Fare", color=NAVY)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=NAVY)
        fig.tight_layout()
        st.pyplot(fig)

    st.divider()
    st.subheader("Finding")
    st.info(
        "In this dataset, female passengers show a higher survival rate than male "
        "passengers, and first-class passengers show a higher survival rate than "
        "third-class passengers. These are observed associations within these 891 "
        "records; the charts do not show that sex or class caused survival."
    )

# ============================================================================
# Tab 3 — Preprocessing
# ============================================================================
with tab_prep:
    st.subheader("Data preparation")

    with st.expander("1 · Missing values", expanded=True):
        st.dataframe(
            pd.DataFrame(
                {
                    "before": raw[["age", "embarked"]].isna().sum(),
                    "after": clean[["age", "embarked"]].isna().sum(),
                }
            ),
            width="stretch",
        )
        st.caption(
            f"age was filled with the median ({age_median:.1f}) and embarked with the "
            f"most frequent category ({embarked_mode}). deck is left out of the "
            f"analysis because {raw['deck'].isna().sum()} of {len(raw)} values are missing."
        )

    with st.expander("2 · Min-Max normalization"):
        st.dataframe(
            clean[["age", "age_norm", "fare", "fare_norm"]].head(10),
            width="stretch",
        )
        st.caption(
            "age_norm and fare_norm = (x - min) / (max - min), so both fall in [0, 1]; "
            "the original age and fare columns are kept."
        )

    with st.expander("3 · Age discretization"):
        st.dataframe(clean[["age", "age_group"]].head(10), width="stretch")
        st.caption(
            "age_group was cut from the original age values: 0-12 Child, 13-18 Teen, "
            "19-35 Young Adult, 36-60 Adult, 61+ Older Adult."
        )
        st.dataframe(
            clean["age_group"].value_counts().reindex(AGE_ORDER).rename("passengers"),
            width="stretch",
        )

    with st.expander("4 · Dummy coding"):
        st.write("Before")
        st.dataframe(clean[["sex", "embarked"]].head(5), width="stretch")
        st.write("After")
        st.dataframe(dummies.head(5), width="stretch")
        st.caption(
            "Dummy variables were created for sex and embarked only. survived is the "
            "target and is not encoded, and the dummy columns are already 0/1 so they "
            "are not normalized."
        )

# ============================================================================
# Tab 4 — Data Explorer
# ============================================================================
with tab_data:
    st.subheader("Data explorer")

    transformed = pd.concat([clean, dummies], axis=1)
    view = st.radio(
        "Select view", ["Raw", "Cleaned", "Transformed"], index=1, horizontal=True
    )
    frame = {"Raw": raw, "Cleaned": clean, "Transformed": transformed}[view]

    default_cols = [c for c in FIELDS + ["age_group"] if c in frame.columns]
    cols = st.multiselect(
        "Choose columns to display", frame.columns.tolist(), default=default_cols
    )

    st.caption(f"{len(filtered):,} rows after filters · showing the {view.lower()} data")
    st.dataframe(
        filtered[cols] if cols else filtered, width="stretch", height=420
    )

    st.divider()
    st.subheader("Load and inspect")

    i1, i2 = st.columns(2)
    with i1:
        st.write(f"**Rows and columns:** {raw.shape[0]} × {raw.shape[1]}")
        st.write("**Column names:**")
        st.write(list(raw.columns))
    with i2:
        st.write("**Data types:**")
        st.dataframe(raw.dtypes.astype(str).rename("dtype"), width="stretch")

    st.write("**First five rows** — `.iloc[:5]`:")
    st.dataframe(raw.iloc[:5], width="stretch")

    st.write("**Missing values per column:**")
    st.dataframe(raw.isna().sum().rename("missing"), width="stretch")

    st.write("**Selected columns** — `.loc[:4, [...]]`:")
    st.dataframe(raw.loc[:4, FIELDS], width="stretch")
