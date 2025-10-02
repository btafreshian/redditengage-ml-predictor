"""Spark training entry point."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_path: str = typer.Argument(..., help="Preprocessed Parquet from preprocess_spark."),
    model_output: str = typer.Argument(..., help="Directory to store the Spark model."),
    metrics_output: str = typer.Option("metrics.json", help="Where to write evaluation metrics."),
    sample_output: str = typer.Option(None, help="Optional path to write a small sample CSV for local experiments."),
    test_fraction: float = typer.Option(0.2, help="Test split proportion."),
    seed: int = typer.Option(42, help="Random seed."),
    max_depth: int = typer.Option(6),
    max_iter: int = typer.Option(80),
) -> None:
    spark = SparkSession.builder.appName("reddit-train").getOrCreate()
    df = spark.read.parquet(input_path)

    train_df, test_df = df.randomSplit([1 - test_fraction, test_fraction], seed=seed)

    assembler = VectorAssembler(
        inputCols=["tfidf_unigram", "tfidf_bigram", "hour", "weekday", "length"],
        outputCol="features",
    )
    gbt = GBTRegressor(featuresCol="features", labelCol="label", maxDepth=max_depth, maxIter=max_iter)
    pipeline = Pipeline(stages=[assembler, gbt])

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    metrics = {
        "rmse": evaluator_rmse.evaluate(predictions),
        "mae": evaluator_mae.evaluate(predictions),
        "test_count": predictions.count(),
    }

    model.write().overwrite().save(model_output)

    Path(metrics_output).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    if sample_output:
        predictions.select("id", "prediction", "label").limit(2000).toPandas().to_csv(sample_output, index=False)

    spark.stop()


if __name__ == "__main__":
    app()
