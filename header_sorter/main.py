import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
import header_sorter.util as util
import header_sorter.features as features
import os.path


if __name__ == '__main__':
    mlp = None

    if not (os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl') and os.path.exists('tags.pkl')):
        print('Fetching data...')
        haaretz_headlines, israel_hayom_headlines = util.read_files()
        haaretz_headlines['label'] = 0
        israel_hayom_headlines['label'] = 1
        all_headlines = np.concatenate((haaretz_headlines['headlines'], israel_hayom_headlines['headlines']))

        print('Vectorizing data...')
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        x = vectorizer.fit_transform(all_headlines)
        df = DataFrame(x.A, columns=vectorizer.get_feature_names())

        print('Processing Haaretz headlines...')
        haaretz_features = features.extract_features(haaretz_headlines)
        print('Processing Israel Hayom headlines...')
        israel_hayom_features = features.extract_features(israel_hayom_headlines)
        print(haaretz_features)
        print(israel_hayom_features)

        for (k, v) in tags.items():
            df[k] = v

        # Split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(df.values,
                                                            np.append(haaretz_headlines['label'],
                                                                      israel_hayom_headlines['label']),
                                                            test_size=0.2,
                                                            random_state=42)

        print('Training model...')
        mlp = MLPClassifier(verbose=True)
        mlp.fit(x_train, y_train)
        joblib.dump(mlp, 'model.pkl')
        for t in tags:
            tags[t] = [0] * len(tags[t])
        joblib.dump(tags, 'tags.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')

        print(mlp.score(x_test, y_test))
    else:
        mlp = joblib.load('model.pkl')
        tags = joblib.load('tags.pkl')
        vectorizer = joblib.load('vectorizer.pkl')

        p = process(["Israel, India pledge to fight evils of terrorism together"])
        print(mlp.predict(sparse.csr_matrix(p.values)))
