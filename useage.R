# restore r environment
#renv::restore()

# load reticulate
library(reticulate)
#Setup new conda environment 
#conda_create("python_proj_default")
# activate the environment for this project
use_condaenv("python_proj_default", required = TRUE)
# Start Python REPL
repl_python()

## install basic packages
#py_install(c("numpy", "pandas", "scipy"), envname = "python_proj_default")

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




