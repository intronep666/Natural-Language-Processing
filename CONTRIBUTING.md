# Contributing Guidelines

Thank you for your interest in contributing to the Natural Language Processing project! We welcome contributions from everyone.

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Process](#development-process)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)
7. [Commit Messages](#commit-messages)

---

## 🤝 Code of Conduct

Please be respectful and professional. We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior
- ✅ Use welcoming and inclusive language
- ✅ Be respectful of different viewpoints
- ✅ Accept constructive criticism gracefully
- ✅ Focus on what is best for the community
- ✅ Show empathy towards others

### Unacceptable Behavior
- ❌ Harassment or discrimination
- ❌ Disrespectful or unwelcoming comments
- ❌ Personal attacks
- ❌ Trolling or inflammatory remarks
- ❌ Any form of abuse

---

## 🚀 Getting Started

### 1. Fork the Repository

```bash
# Go to https://github.com/intronep666/Natural-Language-Processing
# Click the "Fork" button in the top-right
```

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/Natural-Language-Processing.git
cd Natural-Language-Processing
```

### 3. Add Upstream Remote

```bash
git remote add upstream https://github.com/intronep666/Natural-Language-Processing.git
git remote -v  # Verify both origin and upstream
```

### 4. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 💡 How to Contribute

### Types of Contributions

#### 1. **Bug Fixes** 🐛
- Report bugs via GitHub Issues
- Include error messages and reproduction steps
- Fix bugs with clear commit messages

#### 2. **New Features** ✨
- Suggest new features in GitHub Issues first
- Discuss implementation approach
- Add comprehensive examples

#### 3. **Documentation** 📚
- Improve README clarity
- Add missing explanations
- Create tutorials or guides
- Fix typos and grammar

#### 4. **Code Quality** 🔍
- Add comments and docstrings
- Optimize existing code
- Refactor for clarity
- Add type hints

#### 5. **New Practicals** 📖
- Add new NLP technique implementations
- Follow the existing structure and style
- Include detailed comments and explanations
- Add corresponding documentation

#### 6. **Tests** ✅
- Add unit tests
- Improve test coverage
- Test edge cases

---

## 🔄 Development Process

### Step 1: Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch Naming Convention:**
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions

### Step 2: Make Your Changes

```bash
# Edit files
# Test your changes
# Commit regularly with clear messages
```

### Step 3: Keep Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
# Resolve any conflicts if needed
```

### Step 4: Push Your Changes

```bash
git push origin feature/your-feature-name
```

---

## 📝 Pull Request Process

### 1. Create Pull Request

- Go to https://github.com/intronep666/Natural-Language-Processing
- Click "New Pull Request"
- Select your branch
- Fill in the PR template

### 2. PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Other

## Related Issue
Closes #(issue number)

## Testing
Describe tests performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
```

### 3. Respond to Reviews

- Address all feedback promptly
- Request re-review after making changes
- Be respectful of reviewer's time

### 4. Merge

Once approved, your PR will be merged! 🎉

---

## 📋 Coding Standards

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/):

```python
# Good ✓
def calculate_tfidf_score(documents, vocabulary):
    """Calculate TF-IDF scores for documents."""
    scores = {}
    for doc in documents:
        # Process document
        pass
    return scores

# Bad ✗
def calc_tfidf(docs, vocab):
    scores={}
    for d in docs:
        pass
    return scores
```

### Docstrings

Use comprehensive docstrings:

```python
def process_text(text, lowercase=True):
    """
    Process text by tokenization and optional lowercasing.
    
    Args:
        text (str): Input text to process
        lowercase (bool): Whether to convert to lowercase. Default: True
        
    Returns:
        list: List of processed tokens
        
    Example:
        >>> tokens = process_text("Hello World")
        >>> print(tokens)
        ['hello', 'world']
    """
    # Implementation
    pass
```

### Comments

```python
# Use comments for complex logic
# Each line should explain the "why", not the "what"

# Bad: # Increment counter
counter += 1

# Good: # Move to next iteration as current item is invalid
counter += 1
```

### Variable Naming

```python
# Good ✓
max_iterations = 100
is_training = True
vocabulary_size = 5000

# Bad ✗
maxIter = 100
isT = True
vs = 5000
```

### Notebook Conventions

For Jupyter notebooks:

1. **Use descriptive cell titles**
   ```python
   # ### 1. Import Required Libraries
   ```

2. **Add markdown explanations** before code cells

3. **Keep cells focused** - one main task per cell

4. **Add output descriptions**
   ```python
   # Output: Shape of TF-IDF matrix
   print(X.shape)
   ```

5. **Comment complex algorithms**

---

## 💬 Commit Messages

Follow the conventional commit format:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **refactor**: Code refactoring
- **test**: Adding tests
- **style**: Formatting, missing semicolons, etc.
- **perf**: Performance improvements
- **chore**: Build, dependencies, releases

### Examples

```bash
# Good commits ✓
git commit -m "feat(nlp): add BERT embeddings implementation"
git commit -m "fix(tokenizer): handle special characters in preprocessing"
git commit -m "docs: improve GETTING_STARTED guide"
git commit -m "refactor(clustering): optimize K-means algorithm"

# Bad commits ✗
git commit -m "update files"
git commit -m "fixed stuff"
git commit -m "WIP"
```

### Detailed Commit Example

```bash
git commit -m "feat(embedding): implement Word2Vec model with CBOW architecture

- Add Word2Vec class supporting both CBOW and Skip-gram
- Include vector similarity calculation
- Add comprehensive docstrings
- Tested on 20newsgroups dataset

Closes #42"
```

---

## 🔍 Review Process

### What Reviewers Look For

1. **Correctness**: Does the code work as intended?
2. **Quality**: Is the code clean and well-structured?
3. **Documentation**: Are changes documented?
4. **Tests**: Are there appropriate tests?
5. **Standards**: Does it follow our guidelines?

### Response Timeline

- We aim to review PRs within 2-3 days
- Major changes may take longer
- Be patient and respectful

---

## ❓ Questions?

- **Create an Issue**: For feature requests or bug reports
- **Start a Discussion**: For questions and ideas
- **Email**: prexitjoshi@gmail.com

---

## 🎉 Thank You!

Your contributions help make this project better for everyone!

Happy Contributing! 🚀
