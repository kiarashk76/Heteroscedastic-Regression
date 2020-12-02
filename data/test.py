import pickle


with open('data.p', 'rb') as fp:
    data = pickle.load(fp)
# data = {"khar":2}
# with open('data.p', 'wb') as fp:
#     pickle.dump(data, fp)