# restore r environment
#renv::restore()

# load reticulate
library(reticulate)

#Setup new conda environment 
#conda_create("pump_rocket")

# activate the environment for this project
use_condaenv("pump_rocket", required = TRUE)

# Start Python REPL
#repl_python()

#py_install("package", envname = "pump_rocket", pip = TRUE)
# py_install(
#   packages = c(
#     "numpy==1.24.3",           # Array operations
#     "pandas",          # Data manipulation
#     "tqdm",
#     "scikit-learn",    # Ridge classifier and metrics
#     "sktime",          # MiniRocket transformer
#     "matplotlib",      # Plotting
#     "seaborn",         # Enhanced visualizations
#     "scipy"            # Special functions (sigmoid)
#   ),
#   envname = "pump_rocket",
#   pip = TRUE
# )

source_python("01_binary_model.py")

source_python("02_multiclass_model.py")

source_python("03_channel_importance.py")

#source_python("04_channel_importance_multiclass.py")

#source_python("05_run.py")

source_python("06_threshold_comparison.py")

# check conda environments
#conda_list() 

#check current environment
#py_config()

#check python is running
#rint('Hello from Python!')

# Load Python functions from a .py file
#source_python("my_functions.py")

# Run the Python application
#y_run_file("data_processor.py")

# Or source functions and use them in R
#source_python("my_analysis.py")
#result <- process_data("data.csv")

# In Python console:
# >>> import pandas as pd
# >>> df = pd.read_csv("data.csv") 
# >>> exit

# Back in R, access the Python objects:
#py$df  # Access the pandas DataFrame from R
