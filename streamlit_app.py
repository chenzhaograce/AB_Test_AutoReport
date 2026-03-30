import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.proportion import proportions_chisquare
from statsmodels.stats.weightstats import ttest_ind as welch_ttest
import statsmodels.api as sm
import statsmodels.stats.api as sms
import scipy.stats as stats
import math
from sklearn.utils import resample

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

LLM_PROVIDERS = {
    "Gemini (Google)": {
        "models": ["gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"],
        "help": "Free tier available. Get a key at https://aistudio.google.com/apikey",
        "available": GEMINI_AVAILABLE,
        "package": "google-genai",
    },
    "OpenAI": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "help": "Get a key at https://platform.openai.com/api-keys",
        "available": OPENAI_AVAILABLE,
        "package": "openai",
    },
}

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AB Test Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .llm-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 5px solid #667eea;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ─── System Prompt with Embedded Statistical Concepts ────────────────────────

SYSTEM_PROMPT = """You are an expert statistician and data scientist specializing in A/B testing \
and experimentation. Your role is to interpret A/B test results and explain them in clear, \
actionable language that both technical and non-technical stakeholders can understand.

## Your Statistical Knowledge Base

### Probability Distributions

**Normal Distribution (Gaussian)**
The bell-shaped curve defined by mean (μ) and standard deviation (σ). Key properties:
- 68% of data falls within 1σ of the mean, 95% within 2σ, 99.7% within 3σ.
- The Central Limit Theorem states that for large samples (n > 30), sample means follow a \
normal distribution regardless of the underlying data distribution.
- CDF gives P(X ≤ x); for example, P(X ≤ 850) when μ=750, σ=100 is ~0.84.
- This is why parametric tests (t-tests, z-tests) work reliably for large A/B tests.

**Binomial Distribution**
Used for binary outcomes (converted/not converted, clicked/not clicked). Properties:
- n independent trials, each with constant probability p of success.
- PMF gives P(X = k) exactly; CDF gives P(X ≤ k) cumulatively.
- Example: P(exactly 3 purchases out of 10 visitors with p=0.2) ≈ 0.20.
- For large n, the binomial approximates a normal distribution (np > 5 and n(1-p) > 5).
- In A/B testing, conversion rates are binomial: each user either converts (1) or doesn't (0).

### Hypothesis Testing Framework

- **Null Hypothesis (H₀)**: No difference between control and treatment groups.
- **Alternative Hypothesis (H₁)**: There IS a difference between groups.
- **p-value**: Probability of observing results as extreme as the data, assuming H₀ is true. \
It is NOT the probability that H₀ is true.
- **Type I Error (α)**: Rejecting H₀ when it's actually true (false positive). Controlled by \
significance level, typically 0.05.
- **Type II Error (β)**: Failing to reject H₀ when H₁ is true (false negative).
- **Statistical Power (1-β)**: Probability of correctly detecting a real effect. Typically 0.80.

### Statistical Tests Used in A/B Testing

**Z-test** (for proportions / large-sample means):
- z = (x̄₁ - x̄₂ - Δ) / √(s₁²/n₁ + s₂²/n₂)
- One-sided: tests if one group is specifically larger; p = 1 - Φ(z).
- Two-sided: tests if groups differ in either direction; p = 2 × (1 - Φ(|z|)).
- Sample size formula: n = (2(σ₁² + σ₂²)(Z_{α/2})²) / (μ₁ - μ₂)².

**T-test** (parametric, when normality holds):
- Pooled t-test: used when Levene's test confirms equal variances between groups.
- Welch's t-test: used when variances are unequal (more conservative, wider CI).
- For n > 30, t-distribution closely approximates normal distribution.

**Mann-Whitney U test** (non-parametric):
- Used when data is NOT normally distributed (failed Shapiro-Wilk test for small samples).
- Compares medians rather than means; does not assume normal distribution.
- Bootstrap confidence intervals are used alongside for non-parametric CI estimation.

**Chi-square test**:
- Used for Sample Ratio Mismatch (SRM) detection.
- χ² = Σ((observed - expected)² / expected).
- Tests whether observed group sizes match expected proportions (usually 1:1).

**Shapiro-Wilk test**: Tests normality assumption for small samples (n < 100). \
For large samples, CLT ensures approximate normality of means.

**Levene's test**: Tests homogeneity of variance (homoscedasticity) between groups. \
Determines whether to use pooled or Welch's t-test.

### Maximum Likelihood Estimation (MLE)

MLE finds parameter values that maximize the likelihood of observing the given data:
- Likelihood function: L(θ) = Π P(xᵢ | θ) for all observations.
- Log-likelihood: ln(L) = Σ ln(P(xᵢ | θ)) — easier to optimize.
- Take derivative, set to zero, solve for θ.
- For normal distribution: MLE of μ is the sample mean x̄ = (1/n)Σxᵢ.
- For binomial: MLE of p is the sample proportion p̂ = successes/trials.
- In A/B testing, the sample conversion rate IS the MLE estimate of the true rate.
- MLE provides the foundation for estimating CTR, conversion rates, revenue per user, etc.

### Power Analysis & Sample Size

- **Effect Size**: Standardized measure of the difference to detect.
  - For proportions: Cohen's h = 2 × arcsin(√p₁) - 2 × arcsin(√p₂).
  - For means: Cohen's d = (μ₁ - μ₂) / σ_pooled, or (mean × MDE) / std.
- **Sample Size**: Determined by tt_ind_solve_power using effect_size, α, power, and ratio.
- **Test Duration**: (sample_size × (1 + group_ratio)) / avg_daily_unique_users, \
rounded up to complete weeks to avoid day-of-week seasonality bias.
- **MDE**: Minimum Detectable Effect — the smallest meaningful difference worth detecting, \
set by business context (e.g., 5% or 10% relative improvement).

### Validity Checks

**Sample Ratio Mismatch (SRM)**:
- Chi-square test comparing observed vs expected group sizes.
- If p < α: randomization is broken — results CANNOT be trusted.
- Common causes: buggy assignment logic, bot filtering differences, redirect issues.
- This is the FIRST check to run; if SRM fails, stop and investigate.

**Novelty Effect**:
- Users may engage more with something new just because it's new, not because it's better.
- Detected by OLS regression: treatment_conversion ~ time_index.
- Significant negative slope → treatment effect is fading → novelty effect present.
- Mitigation: run test longer, exclude first few days, or use cookie-based holdout.

**AA Test** (pre-test validation):
- Run on historical data BEFORE the experiment.
- If AA test finds significant difference → experimental setup has a problem.
- Common causes: selection bias, logging issues, segment imbalance.
- A passing AA test increases confidence that the experimental infrastructure is sound.

### Confidence Intervals & Lift

- **Absolute Lift**: treatment_mean - control_mean.
- **Relative Lift**: (treatment_mean - control_mean) / control_mean × 100%.
- **Confidence Interval**: Range of plausible values for the true difference.
  - If CI excludes zero → statistically significant.
  - Width indicates measurement precision; narrow = more precise.
  - Parametric CI: from CompareMeans with pooled or unequal variance.
  - Bootstrap CI: resample 1000 times, compute median differences, take 2.5th/97.5th percentiles.
- **CI vs MDE**: If the lower bound of the relative CI exceeds MDE, the test is \
practically significant — ship it. If not, the effect may be real but too small.

## Response Guidelines

1. Start with a clear, one-sentence verdict.
2. Explain what the key numbers mean in plain English.
3. Highlight validity concerns (SRM, novelty, underpowered tests).
4. Provide a clear, actionable recommendation (ship / don't ship / extend / investigate).
5. Explain WHY a specific test was chosen when relevant.
6. Use analogies and simple language; define jargon when you must use it.
7. Be concise but thorough — clarity over completeness.
8. When asked about concepts, draw on the knowledge base above with examples."""


# ─── Analysis Functions ──────────────────────────────────────────────────────

def check_missing(data, id_col, date_col, metric_col):
    df = data[[id_col, date_col, metric_col]]
    missing = df.isnull().sum()
    pct = (df.isnull().sum() / len(df)) * 100
    result = pd.DataFrame({"Total Missing": missing, "Percentage (%)": pct.round(2)})
    return result[result["Total Missing"] > 0]


def check_outliers(data, date_col, metric_col):
    conversions = data.groupby(date_col)[metric_col].sum().reset_index()
    q1 = conversions[metric_col].quantile(0.25)
    q3 = conversions[metric_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = conversions[
        (conversions[metric_col] < lower) | (conversions[metric_col] > upper)
    ]
    return outliers


def calc_effect_size(data, metric_col, binary, mde):
    values = data[metric_col].astype(float)
    if binary:
        avg = values.mean()
        return sm.stats.proportion_effectsize(avg, avg * (1 + mde))
    else:
        return values.mean() * mde / values.std()


def calc_sample_size(effect_size, alpha, power, ratio):
    return tt_ind_solve_power(
        effect_size=effect_size, alpha=alpha, power=power, ratio=ratio
    )


def calc_test_duration(data, date_col, id_col, sample_size, ratio):
    daily_unique = data.groupby(date_col)[id_col].nunique()
    avg_daily = daily_unique.mean()
    if avg_daily == 0:
        return None
    raw_days = sample_size * (1 + ratio) / avg_daily
    return math.ceil(raw_days / 7) * 7


def calc_budget(data, spend_col, id_col, sample_size, ratio):
    total_spend = data[spend_col].sum()
    unique_count = data[id_col].nunique()
    if unique_count == 0:
        return None
    avg_cost = total_spend / unique_count
    return avg_cost * sample_size * (1 + ratio)


def run_aa_test(data, group_col, ctrl_val, trt_val, metric_col):
    control = data[data[group_col] == ctrl_val][metric_col].astype(float)
    treatment = data[data[group_col] == trt_val][metric_col].astype(float)

    if data[metric_col].nunique() == 2:
        _, pval, _ = proportions_chisquare(
            [control.sum(), treatment.sum()],
            nobs=[control.count(), treatment.count()],
        )
    else:
        _, pval, _ = welch_ttest(control, treatment, usevar="unequal")

    return {
        "pvalue": pval,
        "control_mean": control.mean(),
        "treatment_mean": treatment.mean(),
        "control_size": len(control),
        "treatment_size": len(treatment),
    }


def run_srm_test(data, group_col, ctrl_val, trt_val, id_col):
    ab_data = data[data[group_col].isin([ctrl_val, trt_val])]
    observed = ab_data[group_col].value_counts().sort_index().values
    expected = [ab_data.shape[0] * 0.5] * 2
    chi_stat, pval = stats.chisquare(f_obs=observed, f_exp=expected)
    return {
        "pvalue": pval,
        "chi_stat": chi_stat,
        "observed": observed.tolist(),
        "expected": [int(e) for e in expected],
    }


def run_ab_test(data, group_col, ctrl_val, trt_val, metric_col, alpha=0.05):
    control = data[data[group_col] == ctrl_val][metric_col].astype(float)
    treatment = data[data[group_col] == trt_val][metric_col].astype(float)

    ctrl_mean = control.mean()
    trt_mean = treatment.mean()

    desc_trt = sm.stats.DescrStatsW(treatment)
    desc_ctrl = sm.stats.DescrStatsW(control)
    cm = sms.CompareMeans(desc_trt, desc_ctrl)

    if len(control) + len(treatment) < 100:
        sh_ctrl = stats.shapiro(control)[1]
        sh_trt = stats.shapiro(treatment)[1]
        is_normal = (sh_ctrl > alpha) and (sh_trt > alpha)
    else:
        is_normal = True

    is_equal_var = stats.levene(control, treatment)[1] >= alpha

    if is_normal:
        if is_equal_var:
            pval = stats.ttest_ind(control, treatment, equal_var=True)[1]
            lb, ub = cm.tconfint_diff(usevar="pooled")
        else:
            pval = stats.ttest_ind(control, treatment, equal_var=False)[1]
            lb, ub = cm.tconfint_diff(usevar="unequal")
        test_type = "Parametric (t-test)"
        homogeneity = "Equal variance (pooled)" if is_equal_var else "Unequal variance (Welch's)"
    else:
        pval = stats.mannwhitneyu(control, treatment)[1]
        test_type = "Non-Parametric (Mann-Whitney U)"
        homogeneity = "N/A"
        diffs = []
        for _ in range(1000):
            s1 = resample(control.values)
            s2 = resample(treatment.values)
            diffs.append(np.median(s1) - np.median(s2))
        lb = np.percentile(diffs, 2.5)
        ub = np.percentile(diffs, 97.5)

    abs_lift = trt_mean - ctrl_mean
    rel_lift = abs_lift / ctrl_mean if ctrl_mean != 0 else 0
    lower_lift = lb / ctrl_mean if ctrl_mean != 0 else 0
    upper_lift = ub / ctrl_mean if ctrl_mean != 0 else 0

    return {
        "test_type": test_type,
        "homogeneity": homogeneity,
        "is_normal": is_normal,
        "pvalue": pval,
        "control_mean": ctrl_mean,
        "treatment_mean": trt_mean,
        "control_size": len(control),
        "treatment_size": len(treatment),
        "absolute_lift": abs_lift,
        "relative_lift": rel_lift,
        "lb": lb,
        "ub": ub,
        "lower_lift": lower_lift,
        "upper_lift": upper_lift,
    }


def run_novelty_test(data, group_col, ctrl_val, trt_val, date_col, metric_col):
    ab_data = data[data[group_col].isin([ctrl_val, trt_val])]
    per_day = ab_data.groupby([group_col, date_col])[metric_col].mean()

    ctrl_daily = per_day.loc[ctrl_val].reset_index()
    trt_daily = per_day.loc[trt_val].reset_index()

    ctrl_daily.columns = [date_col, "control"]
    trt_daily.columns = [date_col, "treatment"]

    combined = ctrl_daily.merge(trt_daily, on=date_col)
    combined["time_index"] = (combined[date_col] - combined[date_col].min()).dt.days

    X = sm.add_constant(combined["time_index"])
    model = sm.OLS(combined["treatment"], X).fit()

    return {
        "pvalue": model.pvalues["time_index"],
        "rsquared": model.rsquared,
        "coef": model.params["time_index"],
        "daily_data": combined,
    }


# ─── LLM Helper ─────────────────────────────────────────────────────────────

def stream_openai(api_key, context, model="gpt-4o-mini"):
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content


def stream_gemini(api_key, context, model="gemini-2.0-flash"):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content_stream(
        model=model,
        contents=context,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def render_llm_explanation(api_key, provider, context, model, header="AI Interpretation"):
    st.markdown(f"### 🤖 {header}")
    if not api_key:
        st.info("Enter your API key in the sidebar to get AI-powered explanations.")
        return

    provider_info = LLM_PROVIDERS[provider]
    if not provider_info["available"]:
        st.warning(f"Install `{provider_info['package']}` to use {provider}: "
                    f"`pip install {provider_info['package']}`")
        return

    try:
        if provider == "OpenAI":
            st.write_stream(stream_openai(api_key, context, model))
        elif provider == "Gemini (Google)":
            st.write_stream(stream_gemini(api_key, context, model))
    except Exception as e:
        st.error(f"LLM Error: {e}")


# ─── Column Auto-Detection ──────────────────────────────────────────────────

def find_best_index(columns, common_names):
    cols_lower = [c.lower() for c in columns]
    for name in common_names:
        if name.lower() in cols_lower:
            return cols_lower.index(name.lower())
    return 0


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("LLM Settings")
    llm_provider = st.selectbox(
        "Provider",
        list(LLM_PROVIDERS.keys()),
        index=0,
        help="Gemini offers a free tier; OpenAI requires a paid key",
    )
    provider_info = LLM_PROVIDERS[llm_provider]
    api_key = st.text_input(
        f"{llm_provider} API Key",
        type="password",
        help=provider_info["help"],
    )
    llm_model = st.selectbox("Model", provider_info["models"], index=0)

    st.divider()
    st.subheader("Test Parameters")

    mde = st.number_input(
        "Minimum Detectable Effect (MDE)",
        min_value=0.01, max_value=1.0, value=0.10, step=0.01,
        format="%.2f",
        help="Smallest relative improvement worth detecting (e.g. 0.10 = 10%)",
    )
    significance = st.number_input(
        "Significance Level (α)",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        format="%.2f",
        help="Probability of a false positive (Type I error). 0.05 means a 5% chance of concluding there's an effect when there isn't one.",
    )
    power_val = st.number_input(
        "Statistical Power (1 − β)",
        min_value=0.50, max_value=0.99, value=0.80, step=0.05,
        format="%.2f",
        help="Probability of detecting a real effect. 0.80 means an 80% chance of catching a true difference. Higher power requires larger samples.",
    )
    group_ratio = st.number_input(
        "Group Ratio (treatment / control)",
        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        format="%.1f",
        help="1.0 = equal split (50/50)",
    )

# ─── Main Content ────────────────────────────────────────────────────────────

st.title("📊 AB Test Analyzer")
st.caption("Upload your data, run statistical analysis, and get AI-powered interpretations.")

tab_pre, tab_post = st.tabs(["📋 Pre-test Planning", "📊 Post-test Analysis"])

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-TEST TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_pre:
    st.header("Pre-test Planning")
    st.markdown(
        "Upload historical data to calculate the required sample size, test duration, "
        "and validate your experimental setup with an AA test."
    )

    pre_file = st.file_uploader("Upload pre-test data (CSV)", type=["csv"], key="pre_upload")

    if pre_file is not None:
        pre_data = pd.read_csv(pre_file, low_memory=False)
        st.markdown(f"**Preview** — {pre_data.shape[0]:,} rows × {pre_data.shape[1]} columns")
        st.dataframe(pre_data.head(10), use_container_width=True)

        cols = pre_data.columns.tolist()

        with st.expander("🔧 Column Configuration", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                pre_id = st.selectbox(
                    "ID Column", cols, key="pre_id",
                    index=find_best_index(cols, ["impression_id", "user_id", "id", "uid"]),
                    help="Unique identifier for each observation (e.g. user_id, impression_id).",
                )
                pre_date = st.selectbox(
                    "Date Column", cols, key="pre_date",
                    index=find_best_index(cols, ["date", "day", "timestamp", "created_at"]),
                    help="The date/timestamp column used to determine test duration and daily trends.",
                )
            with c2:
                pre_metric = st.selectbox(
                    "Conversion Metric", cols, key="pre_metric",
                    index=find_best_index(cols, ["converted", "conversion", "clicks", "revenue", "action"]),
                    help="The outcome you're measuring — e.g. 'converted' (0/1), 'revenue', or 'clicks'.",
                )
                pre_binary = st.checkbox(
                    "Binary metric (proportion)?", value=True, key="pre_binary",
                    help="Check if the metric is binary (0 or 1, e.g. converted/not). Uncheck for continuous metrics like revenue or time on page.",
                )
            with c3:
                pre_group = st.selectbox(
                    "Group Column (optional)", ["— None —"] + cols, key="pre_group",
                    index=0 if find_best_index(cols, ["group", "variant", "ab_group"]) == 0
                    else find_best_index(["— None —"] + cols, ["group", "variant", "ab_group"]),
                    help="Column that identifies control vs treatment groups. Needed for AA test validation.",
                )
                pre_spend = st.selectbox(
                    "Spend Column (optional)", ["— None —"] + cols, key="pre_spend",
                    help="Cost/spend column for budget estimation. Leave as None if not applicable.",
                )
                pre_experiment = st.selectbox(
                    "Experiment Column (optional)", ["— None —"] + cols, key="pre_exp",
                    help="Column that labels different experiments (e.g. 'AA_test', 'ab_test'). Used to filter for a specific test.",
                )

            has_group = pre_group != "— None —"
            has_spend = pre_spend != "— None —"
            has_experiment = pre_experiment != "— None —"

            if has_group:
                unique_grp = sorted(pre_data[pre_group].dropna().unique().tolist(), key=str)
                gc1, gc2 = st.columns(2)
                with gc1:
                    pre_ctrl = st.selectbox("Control group value", unique_grp, key="pre_ctrl")
                with gc2:
                    remaining = [v for v in unique_grp if v != pre_ctrl]
                    pre_trt = st.selectbox("Treatment group value", remaining, key="pre_trt")

            aa_metric_val = None
            if has_experiment:
                exp_vals = pre_data[pre_experiment].dropna().unique().tolist()
                aa_metric_val = st.selectbox("AA test experiment value", exp_vals, key="pre_aa_val")

        if st.button("🚀 Run Pre-test Analysis", key="pre_run", type="primary", use_container_width=True):
            with st.spinner("Running pre-test analysis..."):
                try:
                    pre_data[pre_date] = pd.to_datetime(pre_data[pre_date])
                except Exception:
                    st.error(f"Could not parse '{pre_date}' as dates. Check the column format.")
                    st.stop()

                # ── Data Quality ──
                st.subheader("1. Data Quality Check")
                dq1, dq2 = st.columns(2)
                with dq1:
                    missing = check_missing(pre_data, pre_id, pre_date, pre_metric)
                    if missing.empty:
                        st.success("No missing values found.")
                    else:
                        st.warning("Missing values detected:")
                        st.dataframe(missing, use_container_width=True)
                with dq2:
                    outliers = check_outliers(pre_data, pre_date, pre_metric)
                    if outliers.empty:
                        st.success("No outliers found.")
                    else:
                        st.warning(f"{len(outliers)} outlier date(s) detected:")
                        st.dataframe(outliers, use_container_width=True)

                # ── Power Analysis ──
                st.subheader("2. Power Analysis & Test Planning")

                analysis_data = pre_data.copy()
                if has_experiment and aa_metric_val:
                    pass  # power analysis on full data

                effect_size = calc_effect_size(analysis_data, pre_metric, pre_binary, mde)
                sample_size = calc_sample_size(effect_size, significance, power_val, group_ratio)
                test_dur = calc_test_duration(analysis_data, pre_date, pre_id, sample_size, group_ratio)
                avg_conversion = analysis_data[pre_metric].astype(float).mean()

                start_date = analysis_data[pre_date].min().date()
                end_date = analysis_data[pre_date].max().date()
                sample_days = (end_date - start_date).days + 1

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Avg Conversion", f"{avg_conversion:.4f}",
                          help="The mean value of your conversion metric across all historical data. This baseline is used to calculate effect size.")
                m2.metric("Sample Size (per group)", f"{sample_size:,.0f}",
                          help="Minimum number of observations needed in EACH group (control and treatment) to reliably detect your MDE at the specified power and significance level.")
                m3.metric("Total Sample Needed", f"{sample_size * (1 + group_ratio):,.0f}",
                          help="Total observations across all groups combined. Equals sample_size × (1 + group_ratio).")
                m4.metric("Test Duration", f"{test_dur} days" if test_dur else "N/A",
                          help="Estimated number of days to collect enough data, based on your historical daily traffic. Rounded up to full weeks to avoid day-of-week bias.")

                budget_val = None
                if has_spend:
                    budget_val = calc_budget(analysis_data, pre_spend, pre_id, sample_size, group_ratio)
                    st.metric("Estimated Budget", f"${budget_val:,.2f}",
                              help="Projected cost based on your historical average cost per observation multiplied by the total sample needed.")

                with st.expander("Details"):
                    st.markdown(f"""
| Parameter | Value |
|---|---|
| Historical data range | {start_date} to {end_date} ({sample_days} days) |
| Effect size (Cohen's d/h) | {effect_size:.4f} |
| MDE | {mde:.0%} |
| Significance level (α) | {significance} |
| Power (1-β) | {power_val} |
| Group ratio | {group_ratio} |
""")

                # ── AA Test ──
                aa_results = None
                if has_group:
                    st.subheader("3. AA Test Validation")

                    if has_experiment and aa_metric_val:
                        aa_data = pre_data[pre_data[pre_experiment] == aa_metric_val]
                    else:
                        aa_data = pre_data[pre_data[pre_group].isin([pre_ctrl, pre_trt])]

                    if len(aa_data) == 0:
                        st.error("No AA test data found after filtering.")
                    else:
                        aa_results = run_aa_test(aa_data, pre_group, pre_ctrl, pre_trt, pre_metric)
                        aa_start = aa_data[pre_date].min().date()
                        aa_end = aa_data[pre_date].max().date()
                        aa_dur = (aa_end - aa_start).days + 1

                        a1, a2, a3 = st.columns(3)
                        a1.metric("AA p-value", f"{aa_results['pvalue']:.4f}",
                                  help="p-value from the AA test. A value above α means no significant difference between groups BEFORE the experiment — this is the expected result. If below α, your setup has a problem.")
                        a2.metric("Control Mean", f"{aa_results['control_mean']:.4f}",
                                  help="Average conversion in the control group during the pre-test period.")
                        a3.metric("Treatment Mean", f"{aa_results['treatment_mean']:.4f}",
                                  help="Average conversion in the treatment group during the pre-test period. Should be close to control mean in a valid AA test.")

                        if aa_results["pvalue"] < significance:
                            st.error(
                                "⚠️ AA test shows significant difference — "
                                "check experimental setup for selection bias or logging issues."
                            )
                        else:
                            st.success(
                                "✅ No significant difference found — "
                                "experimental setup looks valid. Proceed with AB test."
                            )

                        # AA plot
                        ctrl_daily = (
                            aa_data[aa_data[pre_group] == pre_ctrl]
                            .groupby(pre_date)[pre_metric].mean()
                            .reset_index()
                        )
                        trt_daily = (
                            aa_data[aa_data[pre_group] == pre_trt]
                            .groupby(pre_date)[pre_metric].mean()
                            .reset_index()
                        )

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=ctrl_daily[pre_date], y=ctrl_daily[pre_metric],
                            mode="lines+markers", name="Control", line=dict(color="#636EFA"),
                        ))
                        fig.add_trace(go.Scatter(
                            x=trt_daily[pre_date], y=trt_daily[pre_metric],
                            mode="lines+markers", name="Treatment", line=dict(color="#EF553B"),
                        ))
                        fig.update_layout(
                            title="AA Test — Daily Conversion Rate",
                            xaxis_title="Date", yaxis_title="Conversion Rate",
                            template="plotly_white", height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # ── LLM Explanation ──
                st.divider()
                context = f"""Here are the PRE-TEST analysis results for an A/B test. Please interpret them.

## Data Quality
- Dataset: {pre_data.shape[0]:,} rows, {pre_data.shape[1]} columns
- Historical range: {start_date} to {end_date} ({sample_days} days)
- Missing values: {"None" if missing.empty else missing.to_string()}
- Outliers: {"None" if outliers.empty else f"{len(outliers)} date(s)"}

## Test Parameters
- MDE: {mde:.0%}
- Significance level (α): {significance}
- Power (1-β): {power_val}
- Group ratio: {group_ratio}
- Metric type: {"Binary (proportion)" if pre_binary else "Continuous (mean)"}

## Power Analysis
- Average conversion rate: {avg_conversion:.4f}
- Effect size (Cohen's d/h): {effect_size:.4f}
- Required sample size per group: {sample_size:,.0f}
- Total sample needed: {sample_size * (1 + group_ratio):,.0f}
- Recommended test duration: {test_dur} days
{f"- Estimated budget: ${budget_val:,.2f}" if budget_val else "- Budget: not calculated (no spend column)"}
"""
                if aa_results:
                    aa_status = "FAILED — significant difference found" if aa_results["pvalue"] < significance else "PASSED — no significant difference"
                    context += f"""
## AA Test
- AA data range: {aa_start} to {aa_end} ({aa_dur} days)
- Control mean: {aa_results['control_mean']:.4f}, Treatment mean: {aa_results['treatment_mean']:.4f}
- AA p-value: {aa_results['pvalue']:.4f}
- Validation: {aa_status}
"""

                context += "\nPlease provide a complete interpretation and recommendations for running this A/B test."
                render_llm_explanation(api_key, llm_provider, context, llm_model, header="Pre-test Interpretation")


# ═══════════════════════════════════════════════════════════════════════════════
# POST-TEST TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_post:
    st.header("Post-test Analysis")
    st.markdown(
        "Upload experiment data to check validity (SRM, novelty), run the AB test, "
        "and get a clear conclusion with AI-powered explanation."
    )

    post_file = st.file_uploader("Upload post-test data (CSV)", type=["csv"], key="post_upload")

    if post_file is not None:
        post_data = pd.read_csv(post_file, low_memory=False)
        st.markdown(f"**Preview** — {post_data.shape[0]:,} rows × {post_data.shape[1]} columns")
        st.dataframe(post_data.head(10), use_container_width=True)

        cols_p = post_data.columns.tolist()

        with st.expander("🔧 Column Configuration", expanded=True):
            p1, p2, p3 = st.columns(3)
            with p1:
                post_id = st.selectbox(
                    "ID Column", cols_p, key="post_id",
                    index=find_best_index(cols_p, ["impression_id", "user_id", "id", "uid"]),
                    help="Unique identifier for each observation.",
                )
                post_date = st.selectbox(
                    "Date Column", cols_p, key="post_date",
                    index=find_best_index(cols_p, ["date", "day", "timestamp"]),
                    help="Date column used for novelty effect analysis and daily trend charts.",
                )
            with p2:
                post_metric = st.selectbox(
                    "Conversion Metric", cols_p, key="post_metric",
                    index=find_best_index(cols_p, ["converted", "conversion", "clicks", "revenue", "action"]),
                    help="The outcome metric to compare between control and treatment.",
                )
                post_binary = st.checkbox("Binary metric?", value=True, key="post_binary",
                    help="Check if the metric is binary (0/1). Uncheck for continuous metrics like revenue.",
                )
            with p3:
                post_group = st.selectbox(
                    "Group Column", cols_p, key="post_group",
                    index=find_best_index(cols_p, ["group", "variant", "ab_group"]),
                    help="Column identifying which group each observation belongs to (control vs treatment).",
                )
                post_experiment = st.selectbox(
                    "Experiment Column (optional)", ["— None —"] + cols_p, key="post_exp",
                    help="If your data contains multiple experiments, select the column that labels them.",
                )

            unique_grp_p = sorted(post_data[post_group].dropna().unique().tolist(), key=str)
            gp1, gp2 = st.columns(2)
            with gp1:
                post_ctrl = st.selectbox("Control group value", unique_grp_p, key="post_ctrl")
            with gp2:
                remaining_p = [v for v in unique_grp_p if v != post_ctrl]
                post_trt = st.selectbox("Treatment group value", remaining_p, key="post_trt")

            ab_metric_val = None
            post_has_exp = post_experiment != "— None —"
            if post_has_exp:
                exp_vals_p = post_data[post_experiment].dropna().unique().tolist()
                ab_metric_val = st.selectbox("AB test experiment value", exp_vals_p, key="post_ab_val")

        if st.button("🚀 Run Post-test Analysis", key="post_run", type="primary", use_container_width=True):
            with st.spinner("Running post-test analysis..."):
                try:
                    post_data[post_date] = pd.to_datetime(post_data[post_date])
                except Exception:
                    st.error(f"Could not parse '{post_date}' as dates.")
                    st.stop()

                # filter by experiment if specified
                if post_has_exp and ab_metric_val:
                    ab_data = post_data[post_data[post_experiment] == ab_metric_val].copy()
                else:
                    ab_data = post_data[post_data[post_group].isin([post_ctrl, post_trt])].copy()

                if len(ab_data) == 0:
                    st.error("No data found after filtering. Check column configuration.")
                    st.stop()

                test_start = ab_data[post_date].min().date()
                test_end = ab_data[post_date].max().date()
                test_days = (test_end - test_start).days + 1

                # ── Data Quality ──
                st.subheader("1. Data Quality Check")
                dq1, dq2 = st.columns(2)
                with dq1:
                    missing_p = check_missing(ab_data, post_id, post_date, post_metric)
                    if missing_p.empty:
                        st.success("No missing values found.")
                    else:
                        st.warning("Missing values detected:")
                        st.dataframe(missing_p, use_container_width=True)
                with dq2:
                    outliers_p = check_outliers(ab_data, post_date, post_metric)
                    if outliers_p.empty:
                        st.success("No outliers found.")
                    else:
                        st.warning(f"{len(outliers_p)} outlier date(s):")
                        st.dataframe(outliers_p, use_container_width=True)

                # ── SRM Test ──
                st.subheader("2. Sample Ratio Mismatch (SRM)")
                srm = run_srm_test(ab_data, post_group, post_ctrl, post_trt, post_id)
                s1, s2 = st.columns(2)
                s1.metric("SRM p-value", f"{srm['pvalue']:.4f}",
                          help="Sample Ratio Mismatch test. Checks if the split between control and treatment matches the expected ratio (usually 50/50). A low p-value (< α) means the randomization is broken and results cannot be trusted.")
                s2.metric("Chi-square statistic", f"{srm['chi_stat']:.4f}",
                          help="The chi-square test statistic. Measures how far the observed group sizes deviate from the expected sizes. Higher values indicate greater mismatch.")

                st.markdown(f"**Observed**: {srm['observed']}  |  **Expected**: {srm['expected']}")

                srm_pass = srm["pvalue"] >= significance
                if srm_pass:
                    st.success("✅ No sample ratio mismatch — group sizes are balanced.")
                else:
                    st.error(
                        "⚠️ SRM detected — group sizes are significantly imbalanced. "
                        "Investigate the randomization process before trusting other results."
                    )

                # ── AB Test ──
                st.subheader("3. AB Test Results")
                ab = run_ab_test(ab_data, post_group, post_ctrl, post_trt, post_metric, significance)

                r1, r2, r3 = st.columns(3)
                r1.metric("p-value", f"{ab['pvalue']:.4f}",
                          help="The probability of seeing this result (or more extreme) if there were truly no difference between groups. Below α → statistically significant. This is NOT the probability that the treatment doesn't work.")
                r2.metric("Control Mean", f"{ab['control_mean']:.4f}",
                          help="Average value of the conversion metric in the control group (the baseline).")
                r3.metric(
                    "Treatment Mean",
                    f"{ab['treatment_mean']:.4f}",
                    delta=f"{ab['relative_lift']:.2%}",
                    help="Average value of the conversion metric in the treatment group. The delta shows the relative change compared to control.",
                )

                r4, r5, r6 = st.columns(3)
                r4.metric("Absolute Lift", f"{ab['absolute_lift']:.4f}",
                          help="The raw difference: treatment_mean − control_mean. Tells you the actual magnitude of change in the same units as your metric.")
                r5.metric("Relative Lift", f"{ab['relative_lift']:.2%}",
                          help="The percentage change: (treatment − control) / control × 100%. More intuitive for stakeholders — e.g. '5% improvement in conversion rate'.")
                r6.metric("95% CI (relative)", f"[{ab['lower_lift']:.2%}, {ab['upper_lift']:.2%}]",
                          help="95% confidence interval for the relative lift. If this range doesn't cross 0%, the result is statistically significant. If the lower bound exceeds your MDE, the result is also practically significant.")

                with st.expander("Statistical Details"):
                    st.markdown(f"""
| Check | Result |
|---|---|
| Test type | {ab['test_type']} |
| Normal distribution | {"Yes" if ab['is_normal'] else "No"} |
| Variance homogeneity | {ab['homogeneity']} |
| Control size | {ab['control_size']:,} |
| Treatment size | {ab['treatment_size']:,} |
| 95% CI (absolute) | [{ab['lb']:.4f}, {ab['ub']:.4f}] |
""")

                ab_significant = ab["pvalue"] < significance
                if ab_significant:
                    st.success(f"✅ Statistically significant result (p = {ab['pvalue']:.4f} < α = {significance})")
                else:
                    st.warning(f"Result is NOT statistically significant (p = {ab['pvalue']:.4f} ≥ α = {significance})")

                # ── Novelty Effect ──
                st.subheader("4. Novelty Effect Check")
                try:
                    novelty = run_novelty_test(
                        ab_data, post_group, post_ctrl, post_trt, post_date, post_metric
                    )
                    n1, n2 = st.columns(2)
                    n1.metric("Novelty p-value", f"{novelty['pvalue']:.4f}",
                              help="Tests whether the treatment effect changes over time using linear regression. A low p-value (< α) suggests the treatment effect is not stable — likely a novelty or fatigue effect.")
                    n2.metric("Time coefficient", f"{novelty['coef']:.6f}",
                              help="The slope of treatment conversion over time. Negative = performance is declining (novelty wearing off). Positive = performance is improving over time.")

                    if novelty["pvalue"] < significance:
                        st.warning(
                            "⚠️ Novelty effect detected — the treatment effect is changing over time. "
                            "Consider running the test longer or excluding early data."
                        )
                    else:
                        st.success("✅ No novelty effect — treatment performance is stable over time.")

                    daily = novelty["daily_data"]
                    fig_nov = go.Figure()
                    fig_nov.add_trace(go.Scatter(
                        x=daily[post_date], y=daily["control"],
                        mode="lines+markers", name="Control", line=dict(color="#636EFA"),
                    ))
                    fig_nov.add_trace(go.Scatter(
                        x=daily[post_date], y=daily["treatment"],
                        mode="lines+markers", name="Treatment", line=dict(color="#EF553B"),
                    ))
                    fig_nov.update_layout(
                        title="Daily Conversion Rate by Group",
                        xaxis_title="Date", yaxis_title="Conversion Rate",
                        template="plotly_white", height=400,
                    )
                    st.plotly_chart(fig_nov, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not run novelty test: {e}")
                    novelty = None

                # ── Conclusion ──
                st.subheader("5. Conclusion")
                if not srm_pass:
                    st.error(
                        "🚫 **Do not trust the results.** SRM was detected, meaning the "
                        "randomization process is broken. Investigate assignment logic, "
                        "bot filtering, or redirect issues before re-running the test."
                    )
                elif ab_significant:
                    if ab["lower_lift"] >= mde:
                        st.success(
                            f"🎉 **Ship it!** The test shows a **{ab['relative_lift']:.2%}** improvement "
                            f"with CI [{ab['lower_lift']:.2%}, {ab['upper_lift']:.2%}]. "
                            f"The lower bound exceeds MDE ({mde:.0%}), confirming practical significance."
                        )
                    else:
                        st.info(
                            f"📊 The test is statistically significant ({ab['relative_lift']:.2%} lift), "
                            f"but the lower bound of the CI ({ab['lower_lift']:.2%}) is below MDE ({mde:.0%}). "
                            f"Consider re-running with a larger sample to confirm practical significance."
                        )
                else:
                    st.warning(
                        f"📉 **No significant difference** detected (p = {ab['pvalue']:.4f}). "
                        f"The treatment did not demonstrate a meaningful improvement over control."
                    )

                # ── LLM Explanation ──
                st.divider()
                context_post = f"""Here are the POST-TEST analysis results for an A/B test. Please interpret them.

## Data Overview
- Dataset: {ab_data.shape[0]:,} rows
- Test period: {test_start} to {test_end} ({test_days} days)
- Metric type: {"Binary (proportion)" if post_binary else "Continuous (mean)"}
- Missing values: {"None" if missing_p.empty else missing_p.to_string()}
- Outliers: {"None" if outliers_p.empty else f"{len(outliers_p)} date(s)"}

## Test Parameters
- MDE: {mde:.0%}
- Significance level (α): {significance}
- Power (1-β): {power_val}

## SRM Test
- Chi-square p-value: {srm['pvalue']:.4f}
- Observed group sizes: {srm['observed']}
- Expected group sizes: {srm['expected']}
- Result: {"PASSED — no mismatch" if srm_pass else "FAILED — significant mismatch detected"}

## Normality & Homogeneity
- Normal distribution: {"Yes" if ab['is_normal'] else "No"}
- Variance homogeneity: {ab['homogeneity']}
- Test used: {ab['test_type']}

## AB Test Results
- Control mean: {ab['control_mean']:.4f} (n = {ab['control_size']:,})
- Treatment mean: {ab['treatment_mean']:.4f} (n = {ab['treatment_size']:,})
- p-value: {ab['pvalue']:.4f}
- Absolute lift: {ab['absolute_lift']:.4f}
- Relative lift: {ab['relative_lift']:.2%}
- 95% CI (absolute): [{ab['lb']:.4f}, {ab['ub']:.4f}]
- 95% CI (relative): [{ab['lower_lift']:.2%}, {ab['upper_lift']:.2%}]
- Statistically significant: {"Yes" if ab_significant else "No"}
"""
                if novelty:
                    context_post += f"""
## Novelty Effect
- Novelty p-value: {novelty['pvalue']:.4f}
- R-squared: {novelty['rsquared']:.4f}
- Time coefficient: {novelty['coef']:.6f}
- Result: {"Novelty effect DETECTED" if novelty['pvalue'] < significance else "No novelty effect"}
"""

                context_post += f"""
## Conclusion Context
- SRM passed: {srm_pass}
- Statistically significant: {ab_significant}
- Lower CI bound ({ab['lower_lift']:.2%}) vs MDE ({mde:.0%}): {"exceeds MDE" if ab['lower_lift'] >= mde else "below MDE"}

Please provide a thorough interpretation of these results, explain the statistical reasoning, \
and give a clear recommendation on whether to ship the treatment, extend the test, or revert."""

                render_llm_explanation(api_key, llm_provider, context_post, llm_model, header="Post-test Interpretation")

# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "AB Test Analyzer • Statistical analysis powered by scipy & statsmodels • "
    "AI explanations powered by Gemini / OpenAI • "
    "[GitHub](https://github.com/chenzhaograce/AB_Test_AutoReport)"
)
