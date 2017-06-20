from src.pyspark_sc import *

# MLPipeline way
from pyspark.ml import Pipeline
from systemml.mllearn import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 2.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 2.0),
    (4, "b spark who", 1.0),
    (5, "g d a y", 2.0),
    (6, "spark fly", 1.0),
    (7, "was mapreduce", 2.0),
    (8, "e spark program", 1.0),
    (9, "a e c l", 2.0),
    (10, "spark compile", 1.0),
    (11, "hadoop software", 2.0)
], ["id", "text", "label"])
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=20)
lr = LogisticRegression(sqlCtx)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
model = pipeline.fit(training)
test = spark.createDataFrame([
    (12, "spark i j k"),
    (13, "l m n"),
    (14, "mapreduce spark"),
    (15, "apache hadoop")], ["id", "text"])
prediction = model.transform(test)
prediction.show()