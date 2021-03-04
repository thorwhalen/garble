from functools import cached_property
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer(TfidfVectorizer):
    """TfidfVectorizer with extra functionality"""

    @cached_property
    def word_of_idx(self):
        _word_of_idx = {idx: word for word, idx in self.vocabulary_.items()}

        __word_of_idx = np.array([''] * (1 + max(_word_of_idx)), dtype=object)
        __word_of_idx[np.array(list(_word_of_idx.keys()))] = list(_word_of_idx.values())
        return __word_of_idx

    def __call__(self, text):
        return self.transform([text])[0]

    def weight_of_word(self, text):
        sparse_vect = self.transform([text])[0]
        _, idx = sparse_vect.nonzero()
        weights = self.idf_[idx]
        sorting_indices = np.argsort(weights)[::-1]
        return dict(zip(self.word_of_idx[idx][sorting_indices], weights[sorting_indices]))


def word_cloud_gen(text_store, model=None):
    from wordcloud import WordCloud  # pip install wordcloud

    word_cloud_kwargs = dict(width=1000, height=1000)
    if model is None:
        model = TextVectorizer().fit(text_store.values())

    for i, key in enumerate(text_store):
        try:
            text = text_store[key]
            weight_of_word = model.weight_of_word(text)

            wc = WordCloud(**word_cloud_kwargs)
            wc.fit_words(weight_of_word)
            yield wc
        except Exception as e:
            print(f"Error on {i} - {key}: {e}")


url_to_name = lambda url: re.compile(r'http://([^/]+)/?').search(url).group(1)


def compute_and_save_word_clouds(text_store, save_rootdir, model=None, key_to_filename=lambda x: x, ext='.pdf'):
    from wordcloud import WordCloud  # pip install wordcloud

    if model is None:
        model = TextVectorizer().fit(text_store.values())

    if not os.path.isdir(save_rootdir):
        os.mkdir(save_rootdir)

    for i, key in enumerate(text_store):
        try:
            filename = key_to_filename(key)
            save_filepath = os.path.join(save_rootdir, f'{filename}{ext}')
            if not os.path.isfile(save_filepath):
                text = text_store[key]
                weight_of_word = model.weight_of_word(text)

                wc = WordCloud(width=1000, height=1000)
                wc.fit_words(weight_of_word)
                wc.to_file(save_filepath)
        except Exception as e:
            print(f"Error on {i} - {key}: {e}")
