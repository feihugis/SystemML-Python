from src.pyspark_sc import *
import systemml as sml
from systemml import MLContext, dml
import numpy as np

script = """
         source("nn/test/test.dml") as test
         out1 = test::top_k_row()
         out2 = test::top_k()
         """

ml = MLContext(sc)

prog = dml(script)

out = ml.execute(prog)