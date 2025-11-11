import os
from dotenv import load_dotenv
load_dotenv()

# ===========================
# BOARD / STATE DEFINITIONS
# ===========================
BOARD_SIZE = 8
# Planes:
#  - 12 piece planes (6 per color)
#  - 1 en passant
#  - 1 side-to-move
#  - 4 castling rights
#  - 1 repetition / halfmove plane
INPUT_PLANES = (2 * 6) + 1 + 1 + 4 + 1  # = 19
# For most deep RL frameworks, use (channels, height, width)
INPUT_SHAPE = (INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)

# ===========================
# MODEL / NETWORK PARAMETERS
# ===========================
LEARNING_RATE = 3e-4
CONVOLUTION_FILTERS = 256
AMOUNT_OF_RESIDUAL_BLOCKS = 12  # can tune later
BATCH_SIZE = 128
GAMMA = 0.99                # discount factor
TAU = 0.005                 # for target network updates (DQL)

# ===========================
# PPO PARAMETERS
# ===========================
PPO_CLIP_EPSILON = 0.2
PPO_ENTROPY_COEFF = 0.01
PPO_VALUE_COEFF = 0.5
PPO_GAE_LAMBDA = 0.95
PPO_BATCHES_PER_UPDATE = 4
PPO_TIMESTEPS_PER_BATCH = 2048

# ===========================
# DQL PARAMETERS
# ===========================
DQL_LEARNING_RATE = 2.5e-4
DQL_REPLAY_BUFFER_SIZE = 1_000_000
DQL_BATCH_SIZE = 128
DQL_TARGET_UPDATE_FREQ = 10_000   # update every n steps
DQL_EPSILON_START = 1.0
DQL_EPSILON_END = 0.05
DQL_EPSILON_DECAY = 0.9995

# ===========================
# ACTION / OUTPUT DEFINITIONS
# ===========================
# 73 planes: queen (56), knight (8), underpromotions (9)
QUEEN_PLANES = 56
KNIGHT_PLANES = 8
UNDERPROMOTION_PLANES = 9
AMOUNT_OF_PLANES = QUEEN_PLANES + KNIGHT_PLANES + UNDERPROMOTION_PLANES
OUTPUT_SHAPE = (BOARD_SIZE * BOARD_SIZE * AMOUNT_OF_PLANES, 1)
ACTION_SIZE = OUTPUT_SHAPE[0]

# ===========================
# TRAINING CONFIGURATION
# ===========================
MAX_REPLAY_MEMORY = 1_000_000
LOSS_PLOTS_FOLDER = "./plots"
CHECKPOINT_FREQ = 10_000        # steps
SAVE_EVERY_N_GAMES = 100
MODEL_FOLDER = os.environ.get("MODEL_FOLDER", "./models")

# ===========================
# GENERAL SETTINGS
# ===========================
DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"
SEED = int(os.environ.get("SEED", 42))
LOG_INTERVAL = 100
