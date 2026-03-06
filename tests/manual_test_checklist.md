# Manual Testing Checklist - AltScore Streamlit App

## Pre-Deployment Checks

### Environment
- [ ] Python 3.9+ installed
- [ ] All dependencies from requirements.txt installed
- [ ] No errors during `streamlit run app.py`
- [ ] App loads in browser (<10 seconds)

### File Integrity
- [ ] All model files present (lightgbm.pkl, xgboost.pkl, etc.)
- [ ] All visualizations present (8 PNG files)
- [ ] All reports present (3 CSV files)
- [ ] config.py properly configured
- [ ] utils.py has no import errors

## Functional Testing

### Page 1: Home
- [ ] Page loads without errors
- [ ] All 4 metric cards display correctly
- [ ] All 4 innovation cards display correctly
- [ ] System architecture diagram visible
- [ ] Team information displays correctly

### Page 2: Credit Prediction
- [ ] File uploader accepts CSV
- [ ] Sample data generator works
- [ ] Predictions execute successfully
- [ ] Results table displays
- [ ] Download CSV button works
- [ ] Error handling for invalid files works
- [ ] Progress indicators show during processing

### Page 3: Model Comparison
- [ ] Comparison table displays all 4 models
- [ ] ROC curves image loads
- [ ] Confusion matrices image loads
- [ ] Feature importance image loads
- [ ] Feature importance table loads from CSV
- [ ] Novel features highlighted correctly

### Page 4: Fairness Dashboard
- [ ] Metrics load from CSV
- [ ] All 5 attributes displayed
- [ ] Fairness dashboard visualization loads
- [ ] DIR values color-coded correctly (green/red)
- [ ] Expandable sections work for each attribute
- [ ] Age bias warning displays

### Page 5: Counterfactual Explanations
- [ ] File upload works
- [ ] Prediction displays for uploaded applicant
- [ ] Recommendations generate button works
- [ ] All 5 recommendations display
- [ ] Action steps visible for each
- [ ] Timeline visualization displays
- [ ] Download recommendations button works
- [ ] Technical expander works

## UI/UX Testing

### Visual Design
- [ ] Colors consistent with theme
- [ ] Fonts readable (no too small text)
- [ ] Spacing appropriate (not cramped)
- [ ] Images scale properly
- [ ] No text overflow
- [ ] Progress bars work
- [ ] Metrics display properly

### Responsiveness
- [ ] Desktop (1920x1080) works
- [ ] Laptop (1366x768) works
- [ ] Tablet (768x1024) works
- [ ] Mobile (375x667) works
- [ ] Sidebar navigation works on all screens

### Performance
- [ ] Initial page load <5 seconds
- [ ] Navigation between pages instant
- [ ] Predictions <2 seconds for 100 rows
- [ ] Images load quickly (<1 second)
- [ ] No laggy scrolling

## Error Handling

### Edge Cases
- [ ] Empty CSV file
- [ ] CSV with missing columns
- [ ] CSV with 10,000+ rows
- [ ] Invalid feature values (negative, null)
- [ ] Already approved applicant (counterfactual page)
- [ ] Network timeout handling

### Error Messages
- [ ] Clear error messages display
- [ ] Errors don't crash app
- [ ] User can recover from errors
- [ ] No stack traces shown to user

## Browser Compatibility

- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)

## Final Checks

- [ ] No console errors in browser DevTools
- [ ] No broken images
- [ ] All links work
- [ ] Download buttons work
- [ ] All expanders expand/collapse
- [ ] No spelling errors in text
- [ ] Professional appearance
- [ ] Ready for demo
