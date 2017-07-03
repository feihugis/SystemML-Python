from src.pyspark_sc import *
import systemml as sml
from systemml import MLContext, dml
import numpy as np

def threshold():
    script = """
         source("nn/test/test.dml") as test
         out = test::threshold()
         """
    ml = MLContext(sc)
    prog = dml(script)
    out = ml.execute(prog)

def top_k():
    script = """
             source("nn/test/test.dml") as test
             out1 = test::top_k_row()
             out2 = test::top_k()
             """
    ml = MLContext(sc)
    prog = dml(script)
    out = ml.execute(prog)

def top_k2d():
    script = """
             source("nn/test/test.dml") as test
             out = test::top_k2d()
             """
    ml = MLContext(sc)
    ml.setExplain(True)
    prog = dml(script)
    out = ml.execute(prog)

def cross_entropy_loss2d_forward():
    script = """
         source("nn/test/test.dml") as test
         out = test::cross_entropy_loss2d()
         """
    ml = MLContext(sc)
    prog = dml(script)
    out = ml.execute(prog)

def cross_entropy_loss2d_backward():
    script = """
            source("nn/test/grad_check.dml") as grad_check
            out = grad_check::cross_entropy_loss2d()
            """
    ml = MLContext(sc)
    prog = dml(script)
    out = ml.execute(prog)

def softmax2d_backward():
    script = """
            source("nn/test/grad_check.dml") as grad_check
            out = grad_check::softmax2d()
            """
    ml = MLContext(sc)
    prog = dml(script)
    out = ml.execute(prog)

def softmax2d_forward():
    script = """
            source("nn/test/test.dml") as test
            out = test::softmax2d()
            """
    ml = MLContext(sc)
    prog = dml(script)
    out = ml.execute(prog)

if __name__ == "__main__":
    #cross_entropy_loss2d_forward()

    cross_entropy_loss2d_backward()
    #softmax2d_backward()
    #softmax2d_forward()