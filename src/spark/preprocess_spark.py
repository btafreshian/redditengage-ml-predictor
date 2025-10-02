"""Spark preprocessing pipeline for large scale data."""
from __future__ import annotations

from pathlib import Path

import typer
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, NGram, RegexTokenizer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

app = typer.Typer(add_completion=False)


def _read_input(spark: SparkSession, path: str) -> DataFrame:
    file_path = Path(path)
    if file_path.suffix in {".parquet", ".pq"}:
        return spark.read.parquet(path)
    return spark.read.json(path)


def _clean_column(df: DataFrame, text_col: str) -> DataFrame:
    url_pattern = r"https?://\\S+"
    df = df.withColumn(text_col, F.regexp_replace(F.col(text_col), url_pattern, " "))
    df = df.withColumn(text_col, F.regexp_replace(F.col(text_col), "\\s+", " "))
    return df


@app.command()
def main(
    input_path: str = typer.Argument(..., help="Input JSONL or Parquet path."),
    output_path: str = typer.Argument(..., help="Output Parquet location."),
    text_col: str = typer.Option("body", help="Text column name."),
    id_col: str = typer.Option("id", help="Identifier column."),
    time_col: str = typer.Option("created_utc", help="Epoch seconds column."),
    target_col: str = typer.Option("ups", help="Target column."),
    num_features: int = typer.Option(2 ** 15, help="HashingTF feature dimension."),
) -> None:
    spark = SparkSession.builder.appName("reddit-preprocess").getOrCreate()
    df = _read_input(spark, input_path)
    df = _clean_column(df, text_col)

    df = df.withColumn("hour", F.hour(F.from_unixtime(F.col(time_col))))
    df = df.withColumn("weekday", F.dayofweek(F.from_unixtime(F.col(time_col))))
    df = df.withColumn("length", F.length(F.col(text_col)))

    tokenizer = RegexTokenizer(inputCol=text_col, outputCol="tokens", pattern="\\W+", minTokenLength=2)
    hashing = HashingTF(inputCol="tokens", outputCol="tf", numFeatures=num_features)
    ngram = NGram(n=2, inputCol="tokens", outputCol="bigrams")
    hashing2 = HashingTF(inputCol="bigrams", outputCol="tf2", numFeatures=num_features // 2)
    idf = IDF(inputCol="tf", outputCol="tfidf")
    idf2 = IDF(inputCol="tf2", outputCol="tfidf_bi")

    pipeline = Pipeline(stages=[tokenizer, hashing, hashing2, idf, idf2])
    model = pipeline.fit(df)
    transformed = model.transform(df)

    features = transformed.select(
        id_col,
        target_col,
        time_col,
        text_col,
        F.col("hour"),
        F.col("weekday"),
        F.col("length"),
        F.col("tfidf").alias("tfidf_unigram"),
        F.col("tfidf_bi").alias("tfidf_bigram"),
    )

    features = features.withColumnRenamed(target_col, "label")
    features.write.mode("overwrite").parquet(output_path)
    spark.stop()


if __name__ == "__main__":
    app()
