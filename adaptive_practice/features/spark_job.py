import argparse
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from adaptive_practice.data.schema import SESSION_GAP_MS

TRAIN_SCHEMA = T.StructType([
    T.StructField("row_id", T.LongType()),
    T.StructField("timestamp", T.LongType()),
    T.StructField("user_id", T.IntegerType()),
    T.StructField("content_id", T.IntegerType()),
    T.StructField("content_type_id", T.IntegerType()),
    T.StructField("task_container_id", T.IntegerType()),
    T.StructField("user_answer", T.IntegerType()),
    T.StructField("answered_correctly", T.IntegerType()),
    T.StructField("prior_question_elapsed_time", T.DoubleType()),
    T.StructField("prior_question_had_explanation", T.StringType()),
    T.StructField("propensity", T.DoubleType()),
])

SPLITS = {"train": (0, 8), "valid": (8, 9), "test": (9, 10)}


def spark_session(app="adaptive-practice", memory="4g"):
    return (
        SparkSession.builder.appName(app)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def read_inputs(spark, data_dir):
    data = Path(data_dir)
    train = spark.read.csv(str(data / "train.csv"), header=True, schema=TRAIN_SCHEMA)
    questions = spark.read.csv(str(data / "questions.csv"), header=True, inferSchema=True)
    return train, questions


def bundle_table(questions):
    w = Window.orderBy("bundle_id")
    return (
        questions.groupBy("bundle_id")
        .agg(F.first("part").alias("part"), F.count("*").alias("n_questions"))
        .withColumn("bundle_idx", F.row_number().over(w) - 1)
    )


def question_table(questions, bundles):
    return (
        questions.join(bundles.select("bundle_id", "bundle_idx"), "bundle_id")
        .withColumn("tag", F.split(F.coalesce(F.col("tags"), F.lit("0")), " ").getItem(0).cast("int"))
        .select(F.col("question_id").alias("content_id"), "bundle_id", "bundle_idx", "part", "tag")
    )


def history(partition):
    return Window.partitionBy(*partition).orderBy("timestamp").rangeBetween(Window.unboundedPreceding, -1)


def add_split(df):
    bucket = F.pmod(F.xxhash64("user_id"), F.lit(10))
    expr = None
    for name, (lo, hi) in SPLITS.items():
        cond = (bucket >= lo) & (bucket < hi)
        expr = F.when(cond, F.lit(name)) if expr is None else expr.when(cond, F.lit(name))
    return df.withColumn("split", expr)


def containers(q):
    by_user = Window.partitionBy("user_id").orderBy("timestamp")
    c = (
        q.groupBy("user_id", "timestamp")
        .agg(F.first("prior_question_elapsed_time", ignorenulls=True).alias("prior_elapsed"))
        .withColumn("prev_ts", F.lag("timestamp").over(by_user))
        .withColumn("lag_ms", F.col("timestamp") - F.col("prev_ts"))
        .withColumn("new_session", F.when(F.col("prev_ts").isNull() | (F.col("lag_ms") > SESSION_GAP_MS), 1).otherwise(0))
        .withColumn("session_index", F.sum("new_session").over(by_user.rowsBetween(Window.unboundedPreceding, 0)) - 1)
        .withColumn("container_seq", F.row_number().over(by_user) - 1)
    )
    by_session = Window.partitionBy("user_id", "session_index").orderBy("timestamp")
    c = c.withColumn("session_position", F.row_number().over(by_session) - 1)
    nxt = Window.partitionBy("user_id").orderBy("timestamp")
    c = (
        c.withColumn("label_elapsed", F.lead("prior_elapsed").over(nxt))
        .withColumn("next_session", F.lead("session_index").over(nxt))
        .withColumn(
            "label_continue",
            F.when(F.col("next_session") == F.col("session_index"), 1)
            .otherwise(0),
        )
        .withColumn("lag_ms", F.coalesce(F.col("lag_ms"), F.lit(-1)))
        .drop("prev_ts", "new_session", "next_session", "prior_elapsed")
    )
    return c


def build_features(train, questions):
    bundles = bundle_table(questions)
    qmeta = question_table(questions, bundles)

    lectures = (
        train.filter(F.col("content_type_id") == 1)
        .groupBy("user_id", "timestamp").agg(F.count("*").cast("long").alias("n_lectures"))
    )

    q = (
        train.filter(F.col("content_type_id") == 0)
        .join(qmeta, "content_id")
        .withColumn("label_correct", F.col("answered_correctly"))
        .withColumn("prior_explained", F.when(F.lower(F.col("prior_question_had_explanation")) == "true", 1).otherwise(0))
    )
    q = q.join(containers(q), ["user_id", "timestamp"])

    events = (
        q.select("user_id", "timestamp").distinct().withColumn("n_lectures", F.lit(0).cast("long"))
        .unionByName(lectures)
        .withColumn("prior_lectures", F.coalesce(F.sum("n_lectures").over(history(["user_id"])), F.lit(0)))
    )
    last_lec = events.groupBy("user_id", "timestamp").agg(F.max("prior_lectures").alias("prior_lectures"))
    q = q.join(last_lec, ["user_id", "timestamp"])

    for prefix, part in [("prior", ["user_id"]), ("prior_part", ["user_id", "part"]), ("prior_tag", ["user_id", "tag"]),
                         ("session_prior", ["user_id", "session_index"])]:
        w = history(part)
        q = (
            q.withColumn(f"{prefix}_attempts", F.count("label_correct").over(w))
            .withColumn(f"{prefix}_hits", F.coalesce(F.sum("label_correct").over(w), F.lit(0)))
            .withColumn(f"{prefix}_accuracy", (F.col(f"{prefix}_hits") + 1.0) / (F.col(f"{prefix}_attempts") + 2.0))
            .drop(f"{prefix}_hits")
        )

    q = add_split(q)
    stats = (
        q.filter(F.col("split") == "train")
        .groupBy("content_id")
        .agg(
            ((F.sum("label_correct") + 1.0) / (F.count("*") + 2.0)).alias("q_accuracy"),
            F.avg("label_elapsed").alias("q_mean_elapsed"),
            F.count("*").alias("q_attempts"),
        )
    )
    global_acc = q.filter(F.col("split") == "train").agg(F.avg("label_correct")).first()[0] or 0.5
    global_elapsed = q.filter(F.col("split") == "train").agg(F.avg("label_elapsed")).first()[0] or 20000.0
    defaults = {"q_accuracy": float(global_acc), "q_mean_elapsed": float(global_elapsed), "q_attempts": 0}
    qmeta = qmeta.join(stats, "content_id", "left").fillna(defaults)
    q = (
        q.join(stats, "content_id", "left")
        .fillna({"q_accuracy": float(global_acc), "q_mean_elapsed": float(global_elapsed), "q_attempts": 0})
        .withColumn("prior_elapsed", F.coalesce(F.col("prior_question_elapsed_time"), F.lit(-1.0)))
    )
    columns = [
        "split", "user_id", "row_id", "timestamp", "content_id", "bundle_idx", "part", "tag", "container_seq",
        "session_index", "session_position", "lag_ms", "prior_lectures", "prior_attempts", "prior_accuracy",
        "prior_part_attempts", "prior_part_accuracy", "prior_tag_attempts", "prior_tag_accuracy",
        "session_prior_attempts", "session_prior_accuracy", "prior_elapsed", "prior_explained", "q_accuracy",
        "q_mean_elapsed", "q_attempts", "label_correct", "label_elapsed", "label_continue", "propensity",
    ]
    return q.select(*columns), bundles, qmeta


def run(data_dir, out_dir, memory="4g"):
    spark = spark_session(memory=memory)
    train, questions = read_inputs(spark, data_dir)
    features, bundles, qmeta = build_features(train, questions)
    out = Path(out_dir)
    features.repartition("split").write.mode("overwrite").partitionBy("split").parquet(str(out / "interactions"))
    bundles.write.mode("overwrite").parquet(str(out / "bundles"))
    qmeta.write.mode("overwrite").parquet(str(out / "questions"))
    counts = {r["split"]: r["count"] for r in spark.read.parquet(str(out / "interactions")).groupBy("split").count().collect()}
    spark.stop()
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--out", default="data/features")
    parser.add_argument("--memory", default="4g")
    args = parser.parse_args()
    print(run(args.data, args.out, args.memory))


if __name__ == "__main__":
    main()
