def player(prev_play, opponent_history=[]):
    # Initialize histories if empty
    if not opponent_history:
        opponent_history.clear()
        player.player_history = []
    
    # Append opponent's move to history (empty string for first move)
    opponent_history.append(prev_play)
    player.player_history.append("")  # Placeholder for your move

    # Define moves and their counters
    moves = ["R", "P", "S"]
    counters = {"R": "P", "P": "S", "S": "R"}  # Move that beats the key

    # Default move for first game or early games
    if len(opponent_history) <= 5:
        move = moves[len(opponent_history) % 3]  # Cycle R, P, S
        player.player_history[-1] = move
        return move

    # Detect Quincy (fixed sequence: R, R, P, P, S)
    quincy_sequence = ["R", "R", "P", "P", "S"]
    if opponent_history[-5:] == quincy_sequence[:len(opponent_history[-5:])]:
        # Predict Quincy's next move based on sequence
        next_quincy = quincy_sequence[len(opponent_history) % 5]
        move = counters[next_quincy]
        player.player_history[-1] = move
        return move

    # Detect Kris (plays move that beats your last move)
    if len(opponent_history) > 1 and prev_play == counters[player.player_history[-2]]:
        # Kris will play the move that beats our last move
        next_kris = counters[player.player_history[-1]]
        move = counters[next_kris]
        player.player_history[-1] = move
        return move

    # Detect Mrugesh (plays move that beats your most frequent move in last 10)
    if len(opponent_history) > 10:
        recent_moves = player.player_history[-11:-1]
        most_frequent = max(set(recent_moves), key=recent_moves.count)
        mrugesh_counter = counters[most_frequent]
        move = counters[mrugesh_counter]
        player.player_history[-1] = move
        return move

    # Detect Abbey (counters your last move or uses pattern)
    # Assume Abbey plays the move that beats your last move
    if len(opponent_history) > 1:
        abbey_counter = counters[player.player_history[-2]]
        if prev_play == abbey_counter:
            move = counters[counters[player.player_history[-1]]]
            player.player_history[-1] = move
            return move

    # Fallback: Random move
    import random
    move = random.choice(moves)
    player.player_history[-1] = move
    return move