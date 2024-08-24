import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rpath = os.path.abspath('..')
if rpath not in sys.path:
    sys.path.insert(0, rpath)

from scripts.utils import *

st.sidebar.title("Solar Radiation Data Analysis")