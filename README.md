# Fake-News-Classification-Naive-Bayes
NLP project classifying fake and real news using Multinomial Naive Bayes on TF-IDF features. Compares model performance across original, preprocessed, and normalized text data.



# Fake and Real News Classification with Naive Bayes

This project implements a Natural Language Processing (NLP) pipeline to classify news articles as 'Fake' or 'Real' using a **Multinomial Naive Bayes** classifier. It explores the impact of different levels of text preprocessing—namely **Original Text**, **Preprocessed Text (Lowercasing, Tokenization, Stopword Removal)**, and **Normalized Text (Stemming & Lemmatization)**—on model performance, using **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction.

## Getting Started

### Prerequisites

You need Python 3.x installed. The required libraries are listed in the `requirements.txt` file.

### Installation

1.  **Clone the repository:**
    ```bash
  git clone [https://github.com/anamali153/Fake-News-Classification-Naive-Bayes.git](https://github.com/anamali153/Fake-News-Classification-Naive-Bayes.git)
    cd Fake-News-Classification-Naive-Bayes
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK resources:**
    The code requires the `punkt`, `stopwords`, and `wordnet` resources from NLTK. These will be downloaded automatically when the script runs for the first time if they are not present.

## Project Structure

The core logic of the project is contained within a single script (or Jupyter Notebook, if that was the source environment).

| File/Folder | Description |
| :--- | :--- |
| `Fake_and_Real_News_Classification.py` (or `.ipynb`) | Main script containing data download, preprocessing, feature engineering (TF-IDF), model training, and evaluation. |
| `requirements.txt` | Lists all necessary Python dependencies. |
| `.gitignore` | Specifies files and directories to be ignored by Git (e.g., environment files, large datasets). |
| `data/` | Directory for storing the downloaded dataset (created by `kagglehub`). |

## Methodology

1.  **Data Acquisition:** The dataset is downloaded using the `kagglehub` API.
2.  **Data Trimming & Combination:** The first 3,000 rows from both the 'Fake' and 'Real' datasets are taken and combined. Only the `text` and `subject` columns are kept.
3.  **Preprocessing Pipeline:**
    * **Level 1 (Preprocessed):** Lowercasing, URL/Emoji/Non-alphanumeric removal, Tokenization, and Stopword Removal.
    * **Level 2 (Normalized):** Stemming (Porter Stemmer) and Lemmatization (WordNet Lemmatizer) applied to the tokens from Level 1.
4.  **Feature Engineering:** **TF-IDF Vectorization** is applied separately to three text representations:
    * **Original Text**
    * **Level 1: Preprocessed Text**
    * **Level 2: Stemmed/Lemmatized Text**
5.  **Model Training & Evaluation:** A **Multinomial Naive Bayes** classifier is trained and evaluated using a 80/20 train-test split for each of the three feature sets. Performance is reported via the `classification_report`.

## Results and Evaluation

The final part of the script outputs three separate **Classification Reports**, allowing for a direct comparison of how text preprocessing impacts the model's ability to classify the news articles.

*(Note: The actual performance metrics will be printed when running the script.)*
