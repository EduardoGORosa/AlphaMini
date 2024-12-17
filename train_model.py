from neural_network import AlphaZeroNet, train_model
from utils import load_move_mappings

def main():
    move_to_index, index_to_move = load_move_mappings()
    num_moves = len(move_to_index)

    model = AlphaZeroNet(input_channels=13, num_moves=num_moves)

    train_model(
        model=model,
        data_path="self_play_data.pkl",
        move_to_index=move_to_index,
        batch_size=64,
        epochs=10,
        lr=0.001
    )

if __name__ == "__main__":
    main()
