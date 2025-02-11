import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def build_dataframe(folder):

    path = Path(folder)
    df_train = pd.DataFrame(columns=["author"])
    df_test = pd.DataFrame(columns=["author"])
    author_to_id_map = {"kennedy": 0, "johnson": 1}

    def make_df_from_dir(dir_name, df):

        for f in path.glob(f"./{dir_name}/*.txt"):
            with open(f) as fp:
                text = fp.read()

                if dir_name in ("kennedy", "johnson"):
                    df = pd.concat([df, pd.DataFrame({"author": [dir_name], "text": [text]})], ignore_index=True)
                else:

                    author_name = f.stem.split('_')[2]
                    df = pd.concat([df, pd.DataFrame({"author": [author_name], "text": [text]})], ignore_index=True)
        print(df.head)
        return df

    for p in path.iterdir():
        if p.name in ("kennedy", "johnson"):
            df_train = make_df_from_dir(p.name, df_train)
        elif p.name == "unlabeled":
            df_test = make_df_from_dir(p.name, df_test)
    # replace the strings for the author names with numeric codes (0, 1)
    df_train["author"] = df_train["author"].apply(lambda x: author_to_id_map.get(x))
    # do the same for the test data
    df_test["author"] = df_test["author"].apply(lambda x: author_to_id_map.get(x))



    return df_train, df_test


def train_nb(df, alpha=0.1):

    unique_vocabulary = set()
    for text in df["text"]:
        unique_vocabulary.update(text.split())
    vocabulary =  {word: i for i, word in enumerate(sorted(unique_vocabulary))}
    n_docs = df.shape[0]
    n_classes = df["author"].nunique()


    class_counts = df["author"].value_counts().sort_index()
    priors = class_counts / class_counts.sum()

    training_matrix = np.zeros((n_docs, len(vocabulary)))
    for i, text in enumerate(df["text"]):
        for word in text.split():
            if word in vocabulary:
                training_matrix[i, vocabulary[word]] += 1


    word_counts_per_class = np.zeros((n_classes, len(vocabulary)))  
    for i, row in df.iterrows():  
        word_counts_per_class[row["author"]] += training_matrix[i]

    likelihoods =  np.zeros((n_classes, len(vocabulary))) 

    for cl in range(n_classes):
        total_word_count = word_counts_per_class[cl].sum()
        for word_i in range(len(vocabulary)):
            likelihoods[cl, word_i] = (word_counts_per_class[cl, word_i] + alpha) / (total_word_count + alpha * len(vocabulary))

    return vocabulary, priors, likelihoods


def test(df, vocabulary, priors, likelihoods):

    class_predictions = []
    for text in df["text"]:
        test_vector = np.zeros(shape=(len(vocabulary)))

        for word in text.split():
            if word in vocabulary:
                word_i = vocabulary[word]
                test_vector[word_i] += 1 

        preds = np.log(priors) + np.dot(test_vector, np.log(likelihoods.T))

        yhat = np.argmax(preds)
        class_predictions.append(yhat)


    return class_predictions


def sklearn_nb(training_df, test_df):

    
    vectorizer = CountVectorizer()

    vectorizer.fit(training_df["text"])

    training_data = vectorizer.transform(training_df["text"])
    training_data.toarray()

    test_data = vectorizer.transform(test_df["text"])
    test_data.toarray()

    nb_classifier = MultinomialNB()
    nb_classifier.fit(training_data, training_df["author"])

    pred_nb = nb_classifier.predict(test_data)
    return pred_nb


def get_metrics(true, preds):

    accuracy = metrics.accuracy_score(true["author"].tolist(), preds)
    f1_score = metrics.f1_score(true["author"].tolist(), preds, average="weighted")
    conf_matrix = metrics.confusion_matrix(true["author"].tolist(), preds)

    return accuracy, f1_score, conf_matrix


def plot_confusion_matrix(conf_matrix_data, labels):

    plt.title("Confusion matrix")
    axis = sns.heatmap(conf_matrix_data, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, linewidths=0.5)
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set(xlabel="Predicted", ylabel="True")
    plt.savefig("conf.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    args = parser.parse_args()

    training_df, test_df = build_dataframe(args.indir)
    vocabulary, priors, likelihoods = train_nb(training_df)
    print("priors:", priors)
    print("likelihood shape:", len(likelihoods))

    class_predictions = test(test_df, vocabulary, priors, likelihoods)
    print("class_predictions:", class_predictions)

    acc, f1, conf = get_metrics(test_df, class_predictions)
    print("accuracy: ", acc,"\n f1_score: ", f1, "\n conf_matrix: ", conf)

    plot_confusion_matrix(conf, [0, 1])
    

    sklearn_preds = sklearn_nb(training_df, test_df)
    print(sklearn_preds)

    sklearn_metrics = get_metrics(test_df, sklearn_preds)
    print("sklearn metrics (acc,f1,conf): ", sklearn_metrics)
