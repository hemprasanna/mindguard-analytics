# Data Dictionary

## Dataset Overview

This document provides detailed information about all features used in the Suicide Prevention Analytics Platform.

## Raw Features

### Identifier Features

| Feature | Type | Description | Range/Values | Example |
|---------|------|-------------|--------------|---------|
| `post_id` | Integer | Unique identifier for each post | 1 to N | 12345 |

### Text Features

| Feature | Type | Description | Range/Values | Example |
|---------|------|-------------|--------------|---------|
| `text` | String | Content of social media post | Free text | "I feel hopeless..." |
| `post_length` | Integer | Character count of post | 10-500 | 125 |

### Engagement Metrics

| Feature | Type | Description | Range/Values | Example |
|---------|------|-------------|--------------|---------|
| `engagement_rate` | Float | Normalized engagement score (likes, shares, comments) | 0.0 - 1.0 | 0.75 |
| `follower_count` | Integer | Number of account followers | 0 - 10000+ | 1250 |

### Temporal Features

| Feature | Type | Description | Range/Values | Example |
|---------|------|-------------|--------------|---------|
| `time_of_day` | Categorical | When post was created | Morning, Afternoon, Evening, Night | Evening |
| `day_of_week` | Categorical | Day of the week | Mon, Tue, Wed, Thu, Fri, Sat, Sun | Fri |

### Account Features

| Feature | Type | Description | Range/Values | Example |
|---------|------|-------------|--------------|---------|
| `previous_posts_count` | Integer | Total historical posts by user | 0 - 1000+ | 456 |
| `account_age_days` | Integer | Age of account in days | 1 - 3650+ | 730 |

### Sentiment Features

| Feature | Type | Description | Range/Values | Example |
|---------|------|-------------|--------------|---------|
| `sentiment_score` | Float | Overall sentiment polarity | -1.0 (negative) to 1.0 (positive) | -0.65 |

### Target Variable

| Feature | Type | Description | Range/Values | Example |
|---------|------|-------------|--------------|---------|
| `risk_level` | Categorical | Mental health risk classification | Low, Medium, High | High |

## Engineered Features

### Text Analytics Features

| Feature | Type | Description | Range/Values | Source |
|---------|------|-------------|--------------|--------|
| `text_length` | Integer | Length of text in characters | 10-500 | Derived from `text` |
| `word_count` | Integer | Number of words in post | 2-100 | Derived from `text` |
| `polarity` | Float | Sentiment polarity score | -1.0 to 1.0 | TextBlob analysis |
| `subjectivity` | Float | Subjectivity score | 0.0 to 1.0 | TextBlob analysis |
| `contains_negative` | Binary | Contains negative keywords | 0 (No), 1 (Yes) | Keyword detection |

### Temporal Encoded Features

| Feature | Type | Description | Range/Values | Source |
|---------|------|-------------|--------------|--------|
| `time_numeric` | Integer | Numeric encoding of time | 0-3 | Encoded from `time_of_day` |
| `day_numeric` | Integer | Numeric encoding of day | 0-6 | Encoded from `day_of_week` |
| `is_weekend` | Binary | Weekend indicator | 0 (Weekday), 1 (Weekend) | Derived from `day_of_week` |

### Interaction Features

| Feature | Type | Description | Range/Values | Source |
|---------|------|-------------|--------------|--------|
| `engagement_per_follower` | Float | Engagement normalized by followers | 0.0 - 1.0 | `engagement_rate` / `follower_count` |
| `posts_per_day` | Float | Average posts per day | 0.0 - 10.0+ | `previous_posts_count` / `account_age_days` |
| `sentiment_engagement` | Float | Interaction of sentiment and engagement | -1.0 to 1.0 | `sentiment_score` * `engagement_rate` |

### Model Features

| Feature | Type | Description | Range/Values | Source |
|---------|------|-------------|--------------|--------|
| `risk_encoded` | Integer | Encoded target variable | 0 (Low), 1 (Medium), 2 (High) | Encoded from `risk_level` |

## Data Quality Indicators

### Missing Values

Typical missing value patterns in the dataset:

| Feature | Expected Missing % | Handling Strategy |
|---------|-------------------|-------------------|
| `engagement_rate` | 5-10% | Mean/Median/KNN imputation |
| `sentiment_score` | 3-5% | Mean/Median imputation |
| `follower_count` | 1-2% | Median imputation |

### Data Validation Rules

1. **post_length**: Must be > 0 and < 1000
2. **engagement_rate**: Must be between 0.0 and 1.0
3. **account_age_days**: Must be > 0
4. **follower_count**: Must be >= 0
5. **sentiment_score**: Must be between -1.0 and 1.0
6. **risk_level**: Must be one of ['Low', 'Medium', 'High']

## Feature Importance (Typical Rankings)

Based on Random Forest model analysis:

1. **sentiment_score** (0.25) - Most predictive feature
2. **polarity** (0.20) - Strong text-based indicator
3. **contains_negative** (0.15) - Keyword-based flag
4. **engagement_rate** (0.12) - Behavioral indicator
5. **subjectivity** (0.10) - Text characteristic
6. **word_count** (0.08) - Post length indicator
7. **previous_posts_count** (0.05) - Historical behavior
8. **follower_count** (0.03) - Social metric
9. **account_age_days** (0.02) - Account maturity

## Risk Level Definitions

### Low Risk (70% of cases)
- Sentiment score: > 0.2
- No negative keywords
- Normal engagement patterns
- Positive or neutral text content

### Medium Risk (20% of cases)
- Sentiment score: -0.2 to 0.2
- Some concerning language
- Moderate engagement changes
- Mixed emotional content

### High Risk (10% of cases)
- Sentiment score: < -0.2
- Multiple negative keywords
- Significant behavioral changes
- Explicit distress indicators

## Data Sources

1. **Synthetic Generation**: Programmatically generated for demonstration
2. **User Upload**: CSV files with matching schema
3. **Real-time Input**: Manual entry through prediction interface

## Data Processing Pipeline

```
Raw Data
    ↓
Missing Value Imputation
    ↓
Text Analysis (Sentiment, Polarity)
    ↓
Feature Engineering
    ↓
Encoding & Scaling
    ↓
Model Training/Prediction
```

## Usage Notes

- All numeric features are scaled using StandardScaler before model training
- Categorical features are label-encoded
- Text features undergo NLP preprocessing with TextBlob
- Missing values should be handled before feature engineering
- New data must match schema for predictions

## Privacy & Ethics

- **No Personal Identifiers**: Dataset contains no names, emails, or PII
- **Synthetic Data**: Demo data is completely artificial
- **Anonymization**: Real data should be anonymized before upload
- **Consent Required**: Always obtain consent for real-world data collection

## Updates & Versioning

- **Version**: 1.0
- **Last Updated**: 2025
- **Maintained By**: CMSE 830 Project Team

---

For questions about data features or usage, refer to the in-app documentation or README.md.
