from src.pyspark_sc import *
import systemml as sml
from systemml import MLContext, dml
import numpy as np

script = """
         source("nn/test/test.dml") as test
         out = test::top_k2d()
         """

ml = MLContext(sc)

ml.setExplain(True)

prog = dml(script)

out = ml.execute(prog)