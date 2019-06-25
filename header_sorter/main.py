import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import header_sorter.util as util
import header_sorter.features as features
import os.path
from sklearn.neural_network import MLPClassifier
import joblib


if __name__ == '__main__':
    mlp = None

    if not (os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl') and os.path.exists('tags.pkl')):
        print('Fetching data...')
        haaretz_headlines, israel_hayom_headlines = util.read_files()
        haaretz_headlines['label'] = 0
        israel_hayom_headlines['label'] = 1
        all_headlines_np_array = np.concatenate((haaretz_headlines['headlines'], israel_hayom_headlines['headlines']))
        all_headlines = haaretz_headlines.append(israel_hayom_headlines)

        print('Vectorizing data...')
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        x = vectorizer.fit_transform(all_headlines_np_array)
        df = DataFrame(x.A, columns=vectorizer.get_feature_names())

        print('Processing Haaretz headlines...')
        all_features = features.extract_features(all_headlines)

        # Split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(df.values,
                                                            all_features['label'],
                                                            test_size=0.2,
                                                            random_state=42)

        print('Training model...')
        mlp = MLPClassifier(verbose=True)
        mlp.fit(x_train, y_train)
        joblib.dump(mlp, 'model.pkl')

        tags = features.get_tags()
        for t in tags:
            tags[t] = [0] * len(tags[t])
        joblib.dump(tags, 'tags.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')

        print(mlp.score(x_test, y_test))
    # else:
    #     mlp = joblib.load('model.pkl')
    #     tags = joblib.load('tags.pkl')
    #     vectorizer = joblib.load('vectorizer.pkl')
    #
    #     p = process(["Israel, India pledge to fight evils of terrorism together"])
    #     print(mlp.predict(sparse.csr_matrix(p.values)))
