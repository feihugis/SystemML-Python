from src.pyspark_sc import *
import systemml as sml
from systemml import MLContext, dml
import numpy as np

script = """
         source("nn/test/grad_check.dml") as grad_check
         out = grad_check::softmax2d()
         """

ml = MLContext(sc)

prog = dml(script)

out = ml.execute(prog)