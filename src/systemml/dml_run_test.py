from src.pyspark_sc import *
import systemml as sml
from systemml import MLContext, dml, dmlFromResource
import numpy as np

ml = MLContext(sc)

prog = dml("nn/test/run_tests.dml")

out = ml.execute(prog)