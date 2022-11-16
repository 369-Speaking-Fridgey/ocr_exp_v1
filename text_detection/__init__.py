import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detect_model.east.east import EAST as East
from detect_model.ctpn.ctpn import CTPN as Ctpn
# from detect_model.textfuse.textfuse import TEXTFUSE as TextFuse