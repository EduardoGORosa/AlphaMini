# initialize_model.py
import pickle
from neural_network import initialize_and_save_model

def main():
    model_path = "alpha_zero_model.pt"
    move_to_index_path = "move_to_index.pkl"

    # Load move_to_index mapping
    with open(move_to_index_path, "rb") as f:
        move_to_index = pickle.load(f)

    # Initialize and save the model
    initialize_and_save_model(model_path, move_to_index)

if __name__ == "__main__":
    main()
