# --- 1. SETUP & IMPORTS ---

import chess_engine
import chess.svg
import numpy as np
import tensorflow as tf
import json
import time
from IPython.display import display, clear_output

# --- 2. CONFIGURATION ---
MODEL_PATH = "chess_engine_v1.1.keras"
MAP_PATH = "move_map.json"

# --- 3. HELPER FUNCTIONS ---

def board_to_matrix(board: chess_engine.Board):
    """Converts board to 8x8x12 matrix for the model."""
    matrix = np.zeros((8, 8, 12), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        # Layers: 0-5 (White), 6-11 (Black)
        layer = piece.piece_type - 1 + (0 if piece.color == chess_engine.WHITE else 6)
        matrix[row, col, layer] = 1.0
    return matrix

def get_material_score(board):
    """Calculates material difference (White - Black)."""
    scores = {chess_engine.PAWN: 1, chess_engine.KNIGHT: 3, chess_engine.BISHOP: 3, chess_engine.ROOK: 5, chess_engine.QUEEN: 9, chess_engine.KING: 0}
    white_mat = sum([scores[p.piece_type] for p in board.piece_map().values() if p.color == chess_engine.WHITE])
    black_mat = sum([scores[p.piece_type] for p in board.piece_map().values() if p.color == chess_engine.BLACK])
    return white_mat - black_mat

# --- 4. SMART AI LOGIC (BLUNDER CHECKER) ---

def get_smart_move(board, model, int_to_move):
    """
    1. Predicts top 5 moves.
    2. Simulates them to see if they immediately lose material.
    3. Plays the best 'safe' move.
    """
    # Prepare input
    matrix = board_to_matrix(board)
    probs = model.predict(np.expand_dims(matrix, axis=0), verbose=0)[0]
    
    # Get Top 5 Candidate Moves
    top_idxs = probs.argsort()[::-1][:5]
    candidates = []
    
    current_score = get_material_score(board)
    ai_color = board.turn # True for White, False for Black
    
    for idx in top_idxs:
        move_str = int_to_move.get(idx)
        if not move_str: continue
        
        try:
            move = chess_engine.Move.from_uci(move_str)
            if move in board.legal_moves:
                
                # --- SIMULATION START ---
                board.push(move)
                
                # Check: Did we just hang a piece?
                # We look at the opponent's best response (1-ply lookahead)
                is_blunder = False
                
                # Optimization: Only check capturing moves for the opponent
                opponent_responses = list(board.generate_legal_captures())
                
                # If no captures, check standard score
                if not opponent_responses:
                    new_score = get_material_score(board)
                    diff = new_score - current_score
                    if not ai_color: diff = -diff # Flip if Black
                    if diff <= -3: is_blunder = True
                else:
                    # Check if any opponent capture destroys us
                    for resp in opponent_responses:
                        board.push(resp)
                        final_score = get_material_score(board)
                        diff = final_score - current_score
                        if not ai_color: diff = -diff
                        
                        # If opponent can make our score drop by 3+, it's a blunder
                        if diff <= -3: 
                            is_blunder = True
                            board.pop()
                            break
                        board.pop()
                
                board.pop() 
                # --- SIMULATION END ---

                candidates.append((move, probs[idx], is_blunder))
        except: continue
        
    # Sort: Safe moves first (False), then highest confidence
    candidates.sort(key=lambda x: (x[2], -x[1]))
    
    if candidates:
        best_move = candidates[0][0]
        # print debug info
        risk_label = " (RISKY)" if candidates[0][2] else ""
        print(f"AI plays: {best_move.uci()}{risk_label} | Conf: {candidates[0][1]:.1%}")
        return best_move
    
    return None

# --- 5. GAME LOOP ---

def start_game():
    print(" Loading Engine...")
    try:
        # Load Model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Load Vocabulary
        with open(MAP_PATH, "r") as f:
            move_list = json.load(f)
        int_to_move = {i: m for i, m in enumerate(move_list)}
        
        print("Engine Loaded! Game Started.")
    except Exception as e:
        print(f" Error loading files: {e}")
        print("Make sure 'chess_engine.keras' and 'move_map.json' are in the files.")
        return

    board = chess_engine.Board()
    
    while not board.is_game_over():
        clear_output(wait=True)
        display(chess_engine.svg.board(board=board, size=350))
        
        if board.turn == chess_engine.WHITE:
            # --- HUMAN TURN ---
            print("\nYour Turn (White). Enter move (e.g., e2e4):")
            while True:
                uci = input("Move: ").strip()
                if uci == 'quit': return
                try:
                    move = chess_engine.Move.from_uci(uci)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move. Try again.")
                except:
                    print("Invalid format. Use UCI (e.g., e2e4).")
        else:
            # --- AI TURN ---
            print("\n AI is thinking...")
            # Artificial delay to make it feel like an opponent
            time.sleep(0.5) 
            
            best_move = get_smart_move(board, model, int_to_move)
            
            if best_move:
                board.push(best_move)
            else:
                print(" AI Resigns.")
                break

    clear_output(wait=True)
    display(chess_engine.svg.board(board=board, size=350))
    print(f"GAME OVER: {board.result()}")


if __name__ == "__main__":
    start_game()