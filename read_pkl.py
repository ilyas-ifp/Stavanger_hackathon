import pickle

# Path to your .pkl file
file_path = 'data/embeddings/index.pkl'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the content from the file
    data = pickle.load(file)

# Now 'data' contains the object that was stored in the .pkl file
print(data)