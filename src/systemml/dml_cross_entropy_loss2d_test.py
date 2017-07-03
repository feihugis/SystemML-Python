from src.pyspark_sc import *
import systemml as sml
from systemml import MLContext, dml
import numpy as np

script = """
         source("nn/test/test.dml") as test
         out = test::cross_entropy_loss2d()
         """

ml = MLContext(sc)

prog = dml(script)

out = ml.execute(prog)