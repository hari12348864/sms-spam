SMS Spam Classifier — prepared files

1. Put these files in your project folder, alongside your existing spam.csv:
   - train_model.py
   - app.py

2. Activate your existing virtual environment in PyCharm.

3. Install dependencies if needed:
   python -m pip install streamlit pandas scikit-learn nltk

4. Make sure NLTK English stopwords are installed. You already fixed this
   in the previous steps. Test with:
   python -c "from nltk.corpus import stopwords; print(stopwords.words('english')[:10])"

5. Train fresh model files:
   python train_model.py

   This creates/replaces:
   - vectorizer.pkl
   - model.pkl

6. Start the app:
   python -m streamlit run app.py

Important:
- Keep spam.csv in the same folder as train_model.py.
- The CSV must have columns named v1 and v2 (ham/spam label and message).
- The script evaluates a holdout split, then trains the final saved model
  on all cleaned messages.
- Do not use old pickle files after retraining; the new files replace them.
