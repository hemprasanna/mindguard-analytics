# Methodology Documentation

## Project Architecture

### Overview
The Suicide Prevention Analytics Platform uses a multi-layered architecture combining data science, machine learning, and interactive web application development.

## 1. Data Collection & Integration

### Data Sources (Meeting Rubric: 3+ sources)

#### Source 1: Synthetic Data Generation
```python
- Method: Programmatic generation using NumPy
- Size: 1000 samples
- Purpose: Demonstration and testing
- Features: 11 columns with realistic distributions
```

#### Source 2: CSV File Upload
```python
- Method: User upload via Streamlit file_uploader
- Format: CSV with matching schema
- Purpose: Custom dataset analysis
- Validation: Schema checking and data type validation
```

#### Source 3: Real-time Manual Input
```python
- Method: Interactive form input
- Purpose: Single-post prediction
- Features: All required features for prediction
```

## 2. Data Preprocessing Pipeline

### Stage 1: Data Loading & Validation
```python
1. Load data from source
2. Validate schema and data types
3. Check for duplicate records
4. Generate initial quality report
```

### Stage 2: Missing Value Handling

#### Strategy 1: Simple Imputation (Mean)
```python
- Use: Normally distributed features
- Features: engagement_rate, sentiment_score
- Method: sklearn.impute.SimpleImputer(strategy='mean')
```

#### Strategy 2: Simple Imputation (Median)
```python
- Use: Skewed distributions with outliers
- Features: follower_count, previous_posts_count
- Method: sklearn.impute.SimpleImputer(strategy='median')
```

#### Strategy 3: K-Nearest Neighbors (KNN) Imputation
```python
- Use: Features with complex relationships
- Features: All numeric features
- Method: sklearn.impute.KNNImputer(n_neighbors=5)
- Rationale: Preserves feature correlations
```

### Stage 3: Text Preprocessing

#### Sentiment Analysis
```python
Library: TextBlob
Process:
1. Clean text (lowercase, remove special characters)
2. Tokenization
3. Sentiment extraction:
   - Polarity: -1 (negative) to +1 (positive)
   - Subjectivity: 0 (objective) to 1 (subjective)
```

#### Feature Extraction
```python
- text_length: len(text)
- word_count: len(text.split())
- contains_negative: binary flag for keywords
  Keywords: ['hopeless', 'alone', 'end', 'nobody', 'worthless']
```

### Stage 4: Feature Engineering

#### Temporal Features
```python
Encoding Mappings:
- time_of_day: {Morning: 0, Afternoon: 1, Evening: 2, Night: 3}
- day_of_week: {Mon: 0, Tue: 1, ..., Sun: 6}
- is_weekend: Binary (Sat/Sun = 1, else = 0)
```

#### Interaction Features
```python
1. engagement_per_follower = engagement_rate / (follower_count + 1)
   - Avoids division by zero
   - Normalizes engagement by audience size

2. posts_per_day = previous_posts_count / (account_age_days + 1)
   - Measures posting frequency
   - Indicator of account activity level

3. sentiment_engagement = sentiment_score * engagement_rate
   - Captures interaction between mood and reach
   - Positive/negative amplification indicator
```

### Stage 5: Scaling & Normalization

```python
Method: StandardScaler
Formula: z = (x - μ) / σ
Process:
1. Fit on training data
2. Transform both train and test
3. Maintains zero mean, unit variance
```

## 3. Exploratory Data Analysis

### Statistical Analysis

#### Univariate Analysis
```python
Metrics:
- Mean, Median, Mode
- Standard Deviation
- Skewness, Kurtosis
- Quartiles (Q1, Q2, Q3)
- Min, Max, Range
```

#### Bivariate Analysis
```python
Methods:
- Pearson correlation
- Spearman rank correlation
- Chi-square test (categorical)
```

### Visualization Strategy

#### Level 1: Basic Visualizations (Required: 5+)
1. **Pie Chart**: Risk level distribution
2. **Histogram**: Engagement rate distribution
3. **Box Plot**: Numeric feature distributions
4. **Bar Chart**: Temporal patterns
5. **Scatter Plot**: Feature relationships

#### Level 2: Intermediate Visualizations
6. **Heatmap**: Correlation matrix
7. **Violin Plot**: Distribution by category
8. **Scatter Matrix**: Multivariate relationships
9. **Stacked Bar**: Categorical breakdowns
10. **Word Cloud**: Text visualization

#### Level 3: Advanced Visualizations (Above & Beyond)
11. **3D Scatter**: Three-dimensional feature space
12. **Sunburst**: Hierarchical relationships
13. **Parallel Coordinates**: Multivariate patterns
14. **Radar Chart**: Feature profiles
15. **Treemap**: Hierarchical proportions
16. **Sankey Diagram**: Flow analysis

## 4. Machine Learning Pipeline

### Model Selection Rationale

#### Model 1: Random Forest Classifier
```python
Algorithm: Ensemble of decision trees
Parameters:
- n_estimators: 100
- max_depth: None (full growth)
- min_samples_split: 2
- random_state: 42

Advantages:
- Handles non-linear relationships
- Robust to outliers
- Feature importance extraction
- No feature scaling required

Use Case: Primary classification model
```

#### Model 2: Gradient Boosting Classifier
```python
Algorithm: Sequential ensemble with boosting
Parameters:
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 3
- random_state: 42

Advantages:
- High accuracy
- Handles imbalanced data
- Sequential error correction

Use Case: Performance comparison and ensemble
```

#### Model 3: Logistic Regression
```python
Algorithm: Linear classification
Parameters:
- max_iter: 1000
- solver: 'lbfgs'
- multi_class: 'auto'
- random_state: 42

Advantages:
- Interpretable coefficients
- Fast training
- Probability estimates
- Baseline model

Use Case: Benchmark and interpretation
```

### Training Process

#### Data Split
```python
Method: train_test_split
Ratio: 80% training, 20% testing
Stratification: By risk_level
Random State: 42 (reproducibility)
```

#### Cross-Validation
```python
Method: k-Fold Cross-Validation
Folds: 5
Metric: Accuracy
Purpose: Prevent overfitting
```

#### Hyperparameter Tuning
```python
Method: Grid Search (Interactive)
Parameters Tuned:
- Random Forest: n_estimators, max_depth
- Gradient Boosting: learning_rate, n_estimators
- Logistic Regression: C (regularization)

Optimization Metric: Cross-validation accuracy
```

### Evaluation Metrics

#### Classification Metrics
```python
1. Accuracy: (TP + TN) / Total
2. Precision: TP / (TP + FP)
3. Recall: TP / (TP + FN)
4. F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
5. ROC-AUC: Area under ROC curve
```

#### Confusion Matrix Analysis
```python
Structure:
            Predicted
          Low  Med  High
Actual Low   [TP] [FP] [FP]
       Med   [FN] [TP] [FP]
       High  [FN] [FN] [TP]

Interpretation:
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications
- Focus: Minimize High risk false negatives
```

## 5. Model Deployment

### Prediction Pipeline

```python
Input Processing:
1. Receive user input (text + features)
2. Apply text analysis (sentiment, polarity)
3. Engineer features (interactions, temporal)
4. Scale numeric features
5. Predict using trained model
6. Return probabilities + classification

Output:
- Risk level: Low/Medium/High
- Confidence: Probability of prediction
- Feature contributions: SHAP-like importance
```

### Risk Assessment Logic

```python
Threshold Strategy:
- High Risk: P(High) > 0.6 → Immediate intervention
- Medium Risk: 0.3 < P(High) ≤ 0.6 → Monitor closely
- Low Risk: P(High) ≤ 0.3 → Standard monitoring

Decision Support:
- Provide confidence intervals
- Show feature influences
- Recommend actions based on risk level
```

## 6. Application Architecture

### Streamlit Components

#### Session State Management
```python
Purpose: Persist data across interactions
Variables:
- data_loaded: Boolean flag
- processed_data: DataFrame cache
- models: Trained model dictionary

Benefits:
- Avoid redundant computations
- Maintain user workflow state
- Enable multi-page navigation
```

#### Caching Strategy
```python
Decorators:
- @st.cache_data: For data loading/processing
- @st.cache_resource: For model objects

Functions Cached:
- load_sample_data()
- perform_text_analysis()
- handle_missing_values()

Performance Gain: 10-100x faster subsequent runs
```

#### Interactive Elements (15+)
```python
Navigation:
1. Radio buttons (page selection)
2. Sidebar controls
3. File uploader
4. Data table display
5. Dropdown selectors
6. Sliders (hyperparameters)
7. Number inputs
8. Text areas
9. Checkboxes
10. Buttons (primary actions)
11. Tabs (content organization)
12. Expandable sections
13. Metric cards
14. Progress indicators
15. Plotly interactive charts
16. Multiselect widgets
```

## 7. Advanced Techniques

### Text Analytics (NLP)
```python
Techniques:
- Sentiment analysis (TextBlob)
- Keyword extraction
- Word frequency analysis
- N-gram analysis
- TF-IDF vectorization (future enhancement)

Applications:
- Risk keyword detection
- Emotional tone assessment
- Content classification
```

### Ensemble Methods
```python
Strategy: Multiple model combination
Methods:
- Random Forest (bagging)
- Gradient Boosting (boosting)
- Voting Classifier (future)

Benefits:
- Reduced variance
- Improved accuracy
- Robustness to noise
```

### Feature Importance Analysis
```python
Methods:
- Tree-based importance (Gini/entropy)
- Permutation importance
- SHAP values (future)

Insights:
- Which features drive predictions
- Model interpretability
- Feature selection guidance
```

## 8. Quality Assurance

### Testing Strategy
```python
Unit Tests:
- Data validation functions
- Feature engineering correctness
- Model prediction format

Integration Tests:
- End-to-end pipeline
- UI component rendering
- Model training workflow

User Testing:
- Navigation flow
- Error handling
- Performance benchmarks
```

### Performance Optimization
```python
Techniques:
1. Caching expensive operations
2. Lazy loading of models
3. Efficient data structures
4. Vectorized operations (NumPy/Pandas)
5. Plotly (faster than Matplotlib for interactivity)
```

## 9. Ethical Considerations

### Bias Mitigation
```python
Strategies:
- Balanced training data
- Fairness metrics monitoring
- Regular model audits
- Diverse testing scenarios
```

### Privacy Protection
```python
Measures:
- No PII collection
- Synthetic training data
- Session-based storage
- Data anonymization guidelines
```

### Responsible AI
```python
Principles:
- Transparency in decision-making
- Human oversight requirement
- Clear limitations communication
- Professional intervention pathway
```

## 10. Future Enhancements

### Planned Improvements
1. Deep learning models (LSTM, BERT)
2. Real-time data streaming
3. Multi-modal analysis (images, videos)
4. Geographic analysis
5. Longitudinal tracking
6. API integration
7. Mobile optimization
8. Multi-language support

### Research Directions
1. Temporal pattern analysis
2. Network effect modeling
3. Intervention effectiveness tracking
4. Causal inference methods

## References

### Libraries & Frameworks
- Streamlit: Web application framework
- Scikit-learn: Machine learning
- Pandas/NumPy: Data processing
- Plotly: Interactive visualization
- TextBlob: Natural language processing

### Academic Foundations
- Classification algorithms
- Ensemble methods
- NLP techniques
- Risk assessment frameworks
- Mental health crisis indicators

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Maintained By**: CMSE 830 Project Team
