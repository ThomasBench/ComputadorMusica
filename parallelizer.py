import sys
from helpers import treat_tuple, create_driver
import pandas as pd
val=treat_tuple((sys.argv[1],sys.argv[2], sys.argv[3]), create_driver())
print(val)
