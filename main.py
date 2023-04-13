import argparse
import pandas as pd

from classifier.model import DecisionTreeClassifier
from classifier.util import *


def main(args):
    data = pd.read_csv(args.file_path)
    # split
    X, y = data.drop(args.target, axis=1), data.loc[:, args.target]
    split_idx = int(len(y) * args.val_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    if args.load_model:
        model = load_model(args.load_model)
        pred = model.predict(X_test)
    else:
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        if args.save_model:
            save_model(model, args.save_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('val_split')
    parser.add_argument('target')
    parser.add_argument('save_model', required=False)
    parser.add_argument('load_model', required=False)
    args = parser.parse_args()
    main(args)
