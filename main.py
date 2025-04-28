import chess
import numpy as np
import random
import chess.engine
import os
import torch
from ndlinear import NdLinear
import heapq

# Chess grid
ALL_SQUARES = [chess.SQUARE_NAMES[i] for i in range(64)]
PROMOS = ['n', 'b', 'r', 'q']
def all_uci_moves():
    moves = []
    for from_sq in ALL_SQUARES:
        for to_sq in ALL_SQUARES:
            moves.append(from_sq+to_sq)
            for promo in PROMOS:
                moves.append(from_sq+to_sq+promo)
    return moves

MOVE_INDEX = {uci: i for i, uci in enumerate(all_uci_moves())}
INDEX_MOVE = {i: uci for uci, i in MOVE_INDEX.items()}
NUM_MOVES = len(MOVE_INDEX)

# Nd layers with critic actor
class NDPolicyNet:
    def __init__(self, hidden_dim1=128, hidden_dim2=256, hidden_dim3=512):
        self.l1 = NdLinear([16], [hidden_dim1])            
        self.l2 = NdLinear([8*8*hidden_dim1], [hidden_dim2])
        self.l3 = NdLinear([hidden_dim2], [hidden_dim3])
        self.l4 = NdLinear([hidden_dim3], [NUM_MOVES])     
        self.value_head = NdLinear([hidden_dim3], [1])      
        self.parameters = (
            list(self.l1.parameters()) +
            list(self.l2.parameters()) +
            list(self.l3.parameters()) +
            list(self.l4.parameters()) +
            list(self.value_head.parameters())
        )

    def to(self, device):
        self.l1 = self.l1.to(device)
        self.l2 = self.l2.to(device)
        self.l3 = self.l3.to(device)
        self.l4 = self.l4.to(device)
        self.value_head = self.value_head.to(device)
        self.parameters = (
            list(self.l1.parameters()) +
            list(self.l2.parameters()) +
            list(self.l3.parameters()) +
            list(self.l4.parameters()) +
            list(self.value_head.parameters())
        )
        return self

    def forward(self, board_tensor):
        x = self.l1(board_tensor)
        x = torch.relu(x)
        x = x.reshape(-1)
        x = x.unsqueeze(0)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.relu(x)
        policy_logits = self.l4(x).squeeze(0)
        value = self.value_head(x).squeeze(0)
        return policy_logits, value

    def forward(self, board_tensor):
        x = self.l1(board_tensor)
        x = torch.relu(x)
        x = x.reshape(-1)
        x = x.unsqueeze(0)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.relu(x)
        policy_logits = self.l4(x).squeeze(0)
        value = self.value_head(x).squeeze(0)
        return policy_logits, value
    
    #Chess encoding
def board_to_tensor(board):
    planes = np.zeros((8, 8, 16), dtype=np.float32)
    # Piece planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = 'PNBRQK'.index(piece.symbol().upper()) + 6 * int(not piece.color)
            row, col = divmod(square, 8)
            planes[row, col, plane] = 1
    # Whose turn
    planes[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    # Mobility (normalized)
    planes[:, :, 13] = len(list(board.legal_moves)) / 100.0
    # King castled
    planes[:, :, 14] = float(board.has_kingside_castling_rights(board.turn) or
                             board.has_queenside_castling_rights(board.turn))
    # Center control
    cc = [chess.E4, chess.D4, chess.E5, chess.D5]
    planes[:, :, 15] = sum(1 for sq in cc if board.piece_at(sq))
    return torch.from_numpy(planes).float()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = NDPolicyNet(hidden_dim1=128, hidden_dim2=256, hidden_dim3=512).to(device)
optimizer = torch.optim.Adam(policy.parameters, lr=1e-3)


# Discounted Returns 
def compute_discounted_rewards(cache, gamma=0.99):
    discounted = []
    R = 0
    for log_prob, reward, value_pred in reversed(cache):
        R = reward + gamma * R
        R = max(-20.0, min(20.0, R))
        discounted.insert(0, (log_prob, R, value_pred))
    return discounted


# Unsupervised learning model
def reinforce(cache, optimizer, gamma=0.99):
    discounted = compute_discounted_rewards(cache, gamma)
    loss = 0.0
    value_loss = 0.0
    for log_prob, ret, value_pred in discounted:
        log_prob = log_prob.squeeze()
        value_pred = value_pred.squeeze()
        ret_tensor = torch.tensor(ret, device=log_prob.device, dtype=log_prob.dtype).squeeze()
        advantage = ret_tensor - value_pred
        loss += (-log_prob * advantage.detach())
        value_loss += (advantage ** 2)
    total_loss = loss + 0.05 * value_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return float(total_loss.item())

def stockfish_eval(engine, board):
    """Returns pawn eval from Stockfish's perspective."""
    info = engine.analyse(board, chess.engine.Limit(depth=10))
    score = info["score"].white()
    if score.is_mate():
        return 10000 if score.mate() > 0 else -10000
    return score.score() if score.score() is not None else 0


#Epsilon gready move selection
def select_move(policy, board, engine, epsilon=0.1):
    legal_moves = list(board.legal_moves)
    if random.random() < epsilon:
        move = random.choice(legal_moves)
        log_prob = torch.tensor(0.0, device=device)
        value = torch.tensor(0.0, device=device)
        explore = True
        return move, log_prob, value, explore
    tensor = board_to_tensor(board).to(device)
    logits, value = policy.forward(tensor)
    move_indices = [MOVE_INDEX[m.uci()] for m in legal_moves]
    logits = logits[move_indices]
    probs = torch.softmax(logits, dim=0)
    idx = torch.multinomial(probs, 1).item()
    move = legal_moves[idx]
    log_prob = torch.log(probs[idx] + 1e-8)
    explore = False
    return move, log_prob, value, explore

# Reward scale
def calculate_reward(this_eval, last_eval, color='white'):
    eval_diff = this_eval - last_eval if color == 'white' else last_eval - this_eval
    abs_diff = abs(eval_diff)
    if abs_diff < 50:
        scaled_reward = eval_diff / 100.
    elif abs_diff < 200:
        scaled_reward = eval_diff / 20.
    else:
        scaled_reward = -5.0 * (abs_diff / 100.0) if eval_diff < 0 else eval_diff / 10.0
    return float(max(-10.0, min(10.0, scaled_reward)))


#Episode Loop
def run_episode(policy, stockfish_path, color='white', max_moves=200, epsilon=0.1):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    cache = []
    last_eval = stockfish_eval(engine, board)
    move_num = 0
    while not board.is_game_over() and move_num < max_moves:
        if (board.turn == chess.WHITE and color == 'white') or (board.turn == chess.BLACK and color == 'black'):
            move, log_prob, value, explore = select_move(policy, board, engine, epsilon=epsilon)
            board.push(move)
            move_num += 1
            this_eval = stockfish_eval(engine, board)
            reward = calculate_reward(this_eval, last_eval, color)
            cache.append((log_prob, reward, value))
            last_eval = this_eval
        else:
            move = engine.play(board, chess.engine.Limit(time=0.05)).move
            board.push(move)
            last_eval = stockfish_eval(engine, board)
            move_num += 1
    engine.quit()
    # Assign endgame reward
    result = board.result()
    if result == '1-0':
        reward = 2. if color == 'white' else -2.
    elif result == '0-1':
        reward = -2. if color == 'white' else 2.
    else:
        reward = 0.
    cache.append((torch.tensor(0.0, device=device), reward, torch.tensor(0.0, device=device)))
    return cache, result

#Dijkstra style dynamic proority search
def priority_search(board, policy, depth=2):
    class PQEntry:
        def __init__(self, priority, board, move_seq):
            self.priority = priority
            self.board = board.copy()
            self.move_seq = move_seq
        def __lt__(self, other):
            return self.priority < other.priority
    initial_moves = list(board.legal_moves)
    queue = []
    for move in initial_moves:
        board_tmp = board.copy()
        board_tmp.push(move)
        tensor = board_to_tensor(board_tmp).to(device)
        with torch.no_grad():
            _, value = policy.forward(tensor)
        heapq.heappush(queue, PQEntry(-float(value), board_tmp, [move]))
    best_sequence = None
    best_score = float('-inf')
    for _ in range(depth * len(initial_moves)):
        if not queue:
            break
        entry = heapq.heappop(queue)
        if len(entry.move_seq) >= depth:
            if -entry.priority > best_score:
                best_score = -entry.priority
                best_sequence = entry.move_seq
            continue
        for next_move in list(entry.board.legal_moves):
            board_tmp = entry.board.copy()
            board_tmp.push(next_move)
            tensor = board_to_tensor(board_tmp).to(device)
            with torch.no_grad():
                _, value = policy.forward(tensor)
            heapq.heappush(queue, PQEntry(-float(value), board_tmp, entry.move_seq + [next_move]))
    return best_sequence, best_score


stockfish_path = 'C:/Users/Pradip/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe'


# Training Loop
resume_training = True
ckpt_path = 'Chess_engine.pt'

if resume_training and os.path.exists(ckpt_path):
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        policy.l1.load_state_dict(checkpoint['l1'])
        policy.l2.load_state_dict(checkpoint['l2'])
        policy.l3.load_state_dict(checkpoint['l3'])
        policy.l4.load_state_dict(checkpoint['l4'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except KeyError:
            print("Warning:using fresh optimizer.")
        print("Checkpoint loaded!")
    except Exception as e:
        print("Failed to load checkpoint:", e)
        print("Starting from scratch.")
else:
    print("No checkpoint found, starting fresh.")

    epsilon_start = 0.1
    epsilon_end = 0.00001
    epsilon_decay_episodes = 5000


for episode in range(6000):
    cache, result = run_episode(policy, stockfish_path, color='white', epsilon=0.1)
    total_reward = sum(r for _, r, _ in cache)
    loss = reinforce(cache, optimizer)
    print(f"Episode {episode}, Result: {result}, Total reward: {total_reward:.2f}, Loss: {loss:.3f}")


    # Save checkpoint
    if episode % 100 == 0:
        torch.save({
            'l1': policy.l1.state_dict(),
            'l2': policy.l2.state_dict(),
            'l3': policy.l3.state_dict(),
            'l4': policy.l4.state_dict(),
            'value_head': policy.value_head.state_dict(),
            'optimizer': optimizer.state_dict()
        }, 'Chess_engine.pt')
