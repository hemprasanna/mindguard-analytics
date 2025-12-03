# Setup & Deployment Guide

## Quick Start (Local Development)

### 1. Clone or Download Project
```bash
# If using git
git clone <your-repo-url>
cd suicide-prevention-analytics

# Or download and extract ZIP file
```

### 2. Install Python Dependencies
```bash
# Make sure you have Python 3.8+ installed
python --version

# Install required packages
pip install -r requirements.txt

# Download TextBlob corpora (one-time setup)
python -m textblob.download_corpora
```

### 3. Run the Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## GitHub Repository Setup

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it: `suicide-prevention-analytics`
3. Add description: "ML-powered analytics platform for suicide prevention using social media analysis"
4. Choose Public or Private
5. Do NOT initialize with README (we already have one)

### Step 2: Push Files to GitHub

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Suicide Prevention Analytics Platform"

# Add remote repository
git remote add origin https://github.com/YOUR-USERNAME/suicide-prevention-analytics.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Verify Upload

Check that these files are in your repository:
- ✅ app.py
- ✅ requirements.txt
- ✅ README.md
- ✅ .gitignore
- ✅ DATA_DICTIONARY.md
- ✅ METHODOLOGY.md
- ✅ .streamlit/config.toml

## Streamlit Cloud Deployment (Free)

### Prerequisites
- GitHub account with your repository
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Deployment Steps

1. **Sign up for Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "Sign up with GitHub"
   - Authorize Streamlit to access your repositories

2. **Deploy New App**
   - Click "New app"
   - Select your repository: `suicide-prevention-analytics`
   - Main file path: `app.py`
   - Click "Deploy!"

3. **Wait for Deployment**
   - Initial deployment takes 2-5 minutes
   - Streamlit will install dependencies from `requirements.txt`
   - App will be live at: `https://your-app-name.streamlit.app`

4. **Configure Settings (Optional)**
   - Go to app settings
   - Add secrets if needed (not required for this project)
   - Adjust resource limits if needed

### Troubleshooting Deployment

**Issue**: TextBlob corpora not found
**Solution**: Add this to requirements.txt:
```
textblob==0.17.1
```
And add setup.sh:
```bash
#!/bin/bash
python -m textblob.download_corpora
```

**Issue**: Memory limit exceeded
**Solution**: Reduce sample data size in `load_sample_data()` function

**Issue**: Slow performance
**Solution**: Already implemented - app uses @st.cache_data decorators

## Alternative Deployment Options

### Option 1: Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile

# Create runtime.txt
echo "python-3.10.0" > runtime.txt

# Deploy
heroku create your-app-name
git push heroku main
```

### Option 2: Docker
```dockerfile
# Create Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m textblob.download_corpora
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

```bash
# Build and run
docker build -t suicide-prevention-app .
docker run -p 8501:8501 suicide-prevention-app
```

### Option 3: AWS EC2
1. Launch EC2 instance (Ubuntu 20.04)
2. Install Python and dependencies
3. Clone repository
4. Run with nohup or systemd service

## Testing Checklist

Before deployment, verify:

- [ ] All files pushed to GitHub
- [ ] requirements.txt is complete
- [ ] README.md displays correctly
- [ ] App runs locally without errors
- [ ] All pages load correctly
- [ ] Sample data loads successfully
- [ ] File upload works
- [ ] Models train successfully
- [ ] Predictions work correctly
- [ ] All visualizations render
- [ ] No API keys or secrets in code
- [ ] .gitignore excludes unnecessary files

## Post-Deployment Testing

After deployment:

1. **Test Navigation**
   - Click through all pages
   - Verify sidebar works
   - Check tab functionality

2. **Test Data Loading**
   - Load sample data
   - Upload CSV file
   - Verify data displays

3. **Test Features**
   - Run imputation
   - Generate visualizations
   - Train models
   - Make predictions

4. **Test Performance**
   - Check page load times
   - Verify caching works
   - Monitor memory usage

## Maintenance

### Regular Updates
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart app
# Streamlit Cloud auto-deploys on git push
```

### Monitoring
- Check Streamlit Cloud logs for errors
- Monitor app performance
- Review user feedback
- Update models periodically

## Support Resources

### Documentation
- [Streamlit Docs](https://docs.streamlit.io)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- [GitHub Pages](https://pages.github.com)

### Community
- [Streamlit Forum](https://discuss.streamlit.io)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/streamlit)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

### Project-Specific Help
- Check README.md for overview
- Review METHODOLOGY.md for technical details
- See DATA_DICTIONARY.md for data information
- Open GitHub issue for bugs

## Security Best Practices

1. **No Secrets in Code**
   - Use Streamlit secrets management
   - Never commit API keys or passwords

2. **Data Privacy**
   - Don't commit real user data
   - Use synthetic data for demos
   - Anonymize any real datasets

3. **Dependency Updates**
   - Regularly update packages
   - Monitor for security vulnerabilities
   - Use `pip audit` to check dependencies

## Performance Optimization

### Already Implemented
- ✅ Streamlit caching (@st.cache_data)
- ✅ Session state management
- ✅ Efficient data structures (Pandas)
- ✅ Vectorized operations (NumPy)
- ✅ Lazy model loading

### Additional Optimizations
- Consider reducing sample data size for cloud deployment
- Implement pagination for large datasets
- Add loading spinners for long operations
- Optimize visualizations for mobile devices

## Grading Rubric Checklist

Verify your submission includes:

### Base Requirements (80%)
- [x] Three data sources (sample, upload, manual)
- [x] Advanced data cleaning and imputation
- [x] 10+ visualization types
- [x] Statistical analysis
- [x] Feature engineering (text, temporal, interaction)
- [x] Multiple ML models (RF, GB, LR)
- [x] Model evaluation and comparison
- [x] 7 interactive pages
- [x] 15+ Streamlit elements
- [x] Advanced features (caching, session state)
- [x] Professional GitHub repository
- [x] Comprehensive documentation

### Above and Beyond (20%)
- [x] Ensemble methods
- [x] Hyperparameter tuning
- [x] NLP techniques (TextBlob)
- [x] Specialized application (mental health)
- [x] Real-world impact focus
- [x] Publication-quality visualizations

## Contact & Support

For project questions:
- Create GitHub issue
- Contact course instructor
- Check in-app documentation

---

**Good luck with your deployment! 🚀**

Remember: This is a research tool. Always prioritize professional mental health services for real-world applications.
