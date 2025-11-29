# 📋 Quick Reference Guide

A quick lookup guide for common NLP commands, concepts, and notebook structure.

---

## 🚀 Quick Commands

### Installation & Setup
```bash
# Clone repository
git clone https://github.com/intronep666/Natural-Language-Processing.git

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"

# Launch Jupyter
jupyter notebook
```

### Jupyter Shortcuts
| Action | Shortcut |
|--------|----------|
| Run cell | `Shift + Enter` |
| New cell below | `B` |
| New cell above | `A` |
| Delete cell | `D + D` |
| Save notebook | `Ctrl + S` |
| Toggle comment | `Ctrl + /` |

---

## 📚 Practical Notebooks at a Glance

| # | Notebook | Main Topic | Key Skills |
|---|----------|-----------|-----------|
| **01** | Comprehensive Pipeline | Full NLP workflow | Tokenization, POS, NER, Lemmatization |
| **02** | N-Gram Analysis | Word sequences | Unigrams, bigrams, trigrams, probability |
| **03** | Feature Extraction | TF-IDF | Vectorization, importance weighting |
| **04** | Word Embeddings | Semantic vectors | Word2Vec, GloVe, FastText, BERT |
| **05** | Text Classification | Supervised learning | Naïve Bayes, SVM |
| **06** | K-Means Clustering | Unsupervised learning | Document clustering, similarity |
| **07** | POS Tagging | Grammar analysis | Part-of-speech, syntax |
| **08** | LSTM Sentiment | Neural networks | Sequence models, sentiment |
| **09** | Advanced LSTM | Regularization | Dropout, overfitting prevention |
| **10** | Spam Detection | Real-world app | Bag-of-Words, classification |

---

## 🔤 NLP Concepts Quick Reference

### Tokenization
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize("Hello world!")
# Output: ['Hello', 'world', '!']
```

### Lemmatization
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("running")
# Output: "run"
```

### Stemming
```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmer.stem("running")
# Output: "run"
```

### Stop Words
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# Remove: "the", "a", "is", etc.
```

### TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
```

### Word2Vec
```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5)
vector = model.wv['word']
```

### POS Tagging
```python
from nltk import pos_tag, word_tokenize
tokens = word_tokenize("I love NLP")
tags = pos_tag(tokens)
# Output: [('I', 'PRP'), ('love', 'VB'), ('NLP', 'NN')]
```

### Named Entity Recognition
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is in California")
for ent in doc.ents:
    print(f"{ent.text} → {ent.label_}")
```

### Text Classification
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### K-Means Clustering
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)
```

### LSTM Sentiment
```python
from tensorflow.keras.layers import LSTM, Dense, Embedding
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```

---

## 📊 Common NLP Metrics

### Classification
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) - Correctness
- **Recall**: TP / (TP + FN) - Coverage
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Clustering
- **Silhouette Score**: -1 to 1 (higher is better)
- **Davies-Bouldin Index**: Lower is better
- **Calinski-Harabasz Index**: Higher is better

---

## 🔧 Common Import Statements

```python
# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

# spaCy
import spacy
nlp = spacy.load("en_core_web_sm")

# Data Processing
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Word Embeddings
from gensim.models import Word2Vec, FastText
import gensim.downloader as api

# Transformers
from transformers import BertTokenizer, BertModel
```

---

## 📁 File Organization

```
Your Local Folder:
├── 01_Comprehensive_NLP_Pipeline_Linguistic_Analysis.ipynb
├── 02_N_Gram_Analysis_Tokenization_Probability.ipynb
├── ... (practicals 3-10)
├── README.md          ← Start here!
├── GETTING_STARTED.md ← Setup guide
├── CONTRIBUTING.md    ← How to contribute
├── CHANGELOG.md       ← What's new
├── LICENSE            ← MIT License
└── requirements.txt   ← Install dependencies
```

---

## 🎯 Recommended Learning Path

### Beginner (First Time with NLP)
1. Read: README.md overview
2. Do: Setup from GETTING_STARTED.md
3. Run: Practical 01 (Comprehensive Pipeline)
4. Run: Practical 02 (N-Grams)
5. Run: Practical 03 (TF-IDF)

### Intermediate (Some ML knowledge)
6. Run: Practical 04 (Word Embeddings)
7. Run: Practical 05 (Classification)
8. Run: Practical 06 (Clustering)
9. Run: Practical 07 (POS Tagging)

### Advanced (Deep Learning)
10. Run: Practical 08 (LSTM)
11. Run: Practical 09 (Advanced LSTM)
12. Run: Practical 10 (Real-world App)

---

## 🐛 Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | `pip install [package]` |
| NLTK data missing | `nltk.download('[resource]')` |
| spaCy model missing | `python -m spacy download en_core_web_sm` |
| Kernel crashes | Close other apps, reduce batch size |
| Slow training | Use smaller dataset, GPU acceleration |
| Out of memory | Reduce batch size, use smaller model |

---

## 📖 Document Quick Links

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Full project overview |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Setup & installation |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [LICENSE](LICENSE) | MIT License terms |

---

## 🌐 External Resources

### Documentation
- [NLTK Docs](https://www.nltk.org/api/nltk.html)
- [spaCy Docs](https://spacy.io/api)
- [scikit-learn Docs](https://scikit-learn.org/stable/modules/classes.html)
- [TensorFlow/Keras Docs](https://www.tensorflow.org/api_docs)

### Tutorials
- [NLTK Book](https://www.nltk.org/book/)
- [spaCy Course](https://course.spacy.io/)
- [HuggingFace Course](https://huggingface.co/course)

### Communities
- [Stack Overflow NLP](https://stackoverflow.com/questions/tagged/nlp)
- [Reddit r/LanguageTechnology](https://www.reddit.com/r/LanguageTechnology/)
- [HuggingFace Community](https://huggingface.co/community)

---

## ⚡ Pro Tips

1. **Use Virtual Environment**: Keep dependencies isolated
2. **Start Small**: Test with small datasets first
3. **Save Often**: `Ctrl + S` in Jupyter frequently
4. **Read Comments**: Every notebook has detailed comments
5. **Modify Code**: Don't just run - change parameters and experiment!
6. **Document Your Work**: Add notes to notebooks
7. **Version Your Models**: Save trained models for reuse
8. **Use GPU**: If available, speeds up training 10-100x
9. **Monitor Resources**: Watch RAM/CPU usage during training
10. **Ask for Help**: Create GitHub issues or contact maintainer

---

## 📞 Need Help?

- **Questions?** Check [GETTING_STARTED.md](GETTING_STARTED.md) troubleshooting section
- **Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Found a bug?** Open a [GitHub Issue](https://github.com/intronep666/Natural-Language-Processing/issues)
- **Email**: prexitjoshi@gmail.com

---

**Happy Learning! 🚀**

Last Updated: November 2025
