# garble.base

Core text-vectorization tools — a TF-IDF vectorizer with extra conveniences.

### Functions

| `compute_and_save_word_clouds`(text_store, ...)   |    |
|---------------------------------------------------|----|
| `url_to_name`(url)                                |    |
| `word_cloud_gen`(text_store[, model])             |    |

### Classes

| [`TextVectorizer`](#garble.base.TextVectorizer)(\*[, input, encoding, ...])   | TfidfVectorizer with extra functionality   |
|-----------------------------------------------------------------------------------------------|--------------------------------------------|

### *class* garble.base.TextVectorizer(\*, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.float64'>, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

Bases: `TfidfVectorizer`

TfidfVectorizer with extra functionality
