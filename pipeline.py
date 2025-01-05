###############################################
# pipeline.py
###############################################
import os
import datetime
from self_play import self_play_parallel
from neural_network import AlphaZeroNet, load_model_for_inference, train_model
from neural_network import initialize_and_save_model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def automated_training_loop(
    model_path="alpha_zero_model.pt",
    move_to_index_path="move_to_index.pkl",
    num_iterations=2,
    games_per_iteration=2,
    mcts_sims=50,
    temperature=1.0,
    epochs=2,
    batch_size=16,
    lr=0.001,
    num_processes=2
):
    """
    For each iteration:
      1) Generate self-play data in parallel => self_play_data_iter_{it}.pkl
      2) Train the model => alpha_zero_model.pt
      3) Move to next iteration
    """
    print(datetime.datetime.now())
    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = load_model_for_inference(model_path)
    else:
        print("No existing model. Creating new one with policy=20480.")
        initialize_and_save_model(model_path)  # => net with 20480 dimension
        model = load_model_for_inference(model_path)

    for it in range(1, num_iterations+1):
        print(f"\n=== Iteration {it}/{num_iterations} ===")

        # 1) parallel self-play
        data_path = f"self_play_data_iter_{it}.pkl"
        print(f"Generating {games_per_iteration} games with MCTS sims={mcts_sims} => {data_path}")
        self_play_parallel(
            num_games=games_per_iteration,
            mcts_sims=mcts_sims,
            model_path=model_path,
            move_to_index_path=move_to_index_path,
            temperature=temperature,
            output_data_path=data_path,
            num_processes=num_processes
        )

        # 2) train
        print(f"Training on {data_path} ...")
        train_model(
            model=model,
            data_path=data_path,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr
        )

        print(f"Iteration {it} done. Model updated.\n")

    print(datetime.datetime.now())
    print(f"All {num_iterations} iterations done. Final model => {model_path}")

def main():
    automated_training_loop(
        model_path="alpha_zero_model.pt",
        move_to_index_path="move_to_index.pkl",
        num_iterations=1000,
        games_per_iteration=100,
        mcts_sims=800,
        temperature=0.9,
        epochs=20,
        batch_size=16,
        lr=0.001,
        num_processes=16
    )

if __name__ == "__main__":
    main()
