## TODO

---
### Add shaping:
Inside your environment (board_environment.py), you should add rewards for:

✔ Material gain/loss
* +0.05 per capture
* -0.05 if losing a piece

✔ Check or checkmate detection
* +0.02 for giving check
* -0.02 for being put in check

✔ Pawn advancement
* Small positive reward for pushing pawns forward (especially toward promotion).

✔ Penalties:
* small negative reward every move (to avoid infinite play)
* negative reward for repetition
* negative reward for pointless shuffling

---
### Add opponent for training
Option A - random opponent
Option B - heuristic opponent
opponent to optimize quicker training
---
### Add user play environment
Add script so user can interact and play with trained agent

---
### IMPORTANT Add model save / load

---