import pickle


# model utils
def save_model(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_model(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
