# Define True Color (RGB) ANSI codes for Ukraine flag colors
# \033[48;2;R;G;Bm sets the background color
# \033[38;2;R;G;Bm sets the text color (using white/black for contrast)
# \033[0m resets all formatting

# Ukraine Sky Blue: RGB (0, 91, 187) Ukraine Yellow: RGB (255, 213, 0)
# from color-of-victory import BLUE_BG as bb, YELLOW_BG as yb, WHITE_TEXT as wt, BLACK_TEXT as bt, RESET as rset

BLUE_BG = "\033[48;2;0;91;187m"
YELLOW_BG = "\033[48;2;255;213;0m"

# Official Python Emblem Colors (RGB)
PY_BLUE_BG = "\033[48;2;43;108;154m"
PY_YELLOW_BG = "\033[48;2;255;211;32m"

YELLOW_TEXT = "\033[38;2;255;213;0m"
BLUE_TEXT = "\033[38;2;0;91;187m"
WHITE_TEXT = "\033[38;2;255;255;255m"
BLACK_TEXT = "\033[38;2;0;0;0m"
RESET = "\033[0m"

# Print the colored blocks
#print(f"{BL_B}{WH_T} 🐍 PYTHON   {RSET}")
#print(f"{YE_B}{BL_T}  ENVIRONMENT {RSET}")
      
# Print a Python emblem-themed CLI block
#print(f"{PY_BLUE_BG}{WHITE_TEXT}  🐍 PYTHON   {RSET}")
#print(f"{PY_YELLOW_BG}{BLACK_TEXT}  ENVIRONMENT {RSET}")