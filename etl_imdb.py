from pathlib import Path
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "imdb-data"
RESULTS_DIR = BASE_DIR / "imdb-results"

# Схеми відповідають структурі файлів IMDb Datasets.
SCHEMA_AKAS = T.StructType([
    T.StructField("titleId", T.StringType(), False),
    T.StructField("ordering", T.IntegerType(), False),
    T.StructField("title", T.StringType(), True),
    T.StructField("region", T.StringType(), True),
    T.StructField("language", T.StringType(), True),
    T.StructField("types", T.StringType(), True),
    T.StructField("attributes", T.StringType(), True),
    T.StructField("isOriginalTitle", T.IntegerType(), True),
])

SCHEMA_NAME = T.StructType([
    T.StructField("nconst", T.StringType(), False),
    T.StructField("primaryName", T.StringType(), True),
    T.StructField("birthYear", T.IntegerType(), True),
    T.StructField("deathYear", T.IntegerType(), True),
    T.StructField("primaryProfession", T.StringType(), True),
    T.StructField("knownForTitles", T.StringType(), True),
])

SCHEMA_BASICS = T.StructType([
    T.StructField("tconst", T.StringType(), False),
    T.StructField("titleType", T.StringType(), True),
    T.StructField("primaryTitle", T.StringType(), True),
    T.StructField("originalTitle", T.StringType(), True),
    T.StructField("isAdult", T.IntegerType(), True),
    T.StructField("startYear", T.IntegerType(), True),
    T.StructField("endYear", T.IntegerType(), True),
    T.StructField("runtimeMinutes", T.IntegerType(), True),
    T.StructField("genres", T.StringType(), True),
])

SCHEMA_PRINCIPALS = T.StructType([
    T.StructField("tconst", T.StringType(), False),
    T.StructField("ordering", T.IntegerType(), False),
    T.StructField("nconst", T.StringType(), False),
    T.StructField("category", T.StringType(), True),
    T.StructField("job", T.StringType(), True),
    T.StructField("characters", T.StringType(), True),
])

SCHEMA_EPISODE = T.StructType([
    T.StructField("tconst", T.StringType(), False),
    T.StructField("parentTconst", T.StringType(), True),
    T.StructField("seasonNumber", T.IntegerType(), True),
    T.StructField("episodeNumber", T.IntegerType(), True),
])

SCHEMA_RATINGS = T.StructType([
    T.StructField("tconst", T.StringType(), False),
    T.StructField("averageRating", T.FloatType(), True),
    T.StructField("numVotes", T.IntegerType(), True),
])


def read_data(spark, file_name: str, schema: T.StructType):
    """Зчитує TSV.GZ, використовуючи схему та трактуючи \\N як NULL."""
    path = DATA_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"Не знайдено {path}. Завантажте відповідний файл IMDb у каталог imdb-data."
        )
    return (
        spark.read
        .option("header", True)
        .option("sep", "\\t")
        .option("nullValue", "\\N")
        .schema(schema)
        .csv(str(path))
    )


def write_data(df, task_name: str):
    """Зберігає результат у TSV з заголовком у власний каталог."""
    output_path = RESULTS_DIR / task_name
    (
        df.write.mode("overwrite")
        .option("header", True)
        .option("sep", "\\t")
        .option("nullValue", "\\N")
        .csv(str(output_path))
    )
    print(f"Результат збережено: {output_path}")


def show_and_save(df, task_name: str, rows=10):
    print(f"\\n===== {task_name} =====")
    df.show(rows, truncate=False)
    write_data(df, task_name)


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("IMDb ETL Laboratory 4")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # EXTRACT: завантаження джерел.
    akas = read_data(spark, "title.akas.tsv.gz", SCHEMA_AKAS)
    names = read_data(spark, "name.basics.tsv.gz", SCHEMA_NAME)
    basics = read_data(spark, "title.basics.tsv.gz", SCHEMA_BASICS)
    principals = read_data(spark, "title.principals.tsv.gz", SCHEMA_PRINCIPALS)
    episodes = read_data(spark, "title.episode.tsv.gz", SCHEMA_EPISODE)
    ratings = read_data(spark, "title.ratings.tsv.gz", SCHEMA_RATINGS)

    print("Схема title.basics:")
    basics.printSchema()
    print("Кількість записів title.basics:", basics.count())

    # 1. Усі назви, доступні в українському регіоні UA.
    task1 = (
        akas.filter(F.col("region") == "UA")
        .filter(F.col("title").isNotNull())
        .select(
            F.col("titleId"),
            F.col("title").alias("title_in_Ukrainian"),
            F.col("language")
        )
        .dropDuplicates()
    )
    show_and_save(task1, "task1_ukrainian_titles")

    # 2. Люди, народжені у XIX столітті (1801–1900 включно).
    task2 = (
        names.filter(F.col("birthYear").between(1801, 1900))
        .select("nconst", "primaryName", "birthYear", "deathYear")
        .orderBy("birthYear", "primaryName")
    )
    show_and_save(task2, "task2_people_19th_century")

    # 3. Фільми тривалістю понад 120 хвилин; беремо оригінальну назву з akas.
    long_movies = (
        basics.filter((F.col("titleType") == "movie") & (F.col("runtimeMinutes") > 120))
        .select("tconst", "runtimeMinutes")
    )
    original_titles = (
        akas.filter(F.col("isOriginalTitle") == 1)
        .select(F.col("titleId").alias("akas_titleId"), F.col("title").alias("original_title"))
        .dropDuplicates(["akas_titleId"])
    )
    task3 = (
        long_movies.join(original_titles, long_movies.tconst == original_titles.akas_titleId, "inner")
        .select("tconst", "original_title", "runtimeMinutes")
        .filter(F.col("original_title").isNotNull())
        .orderBy(F.col("runtimeMinutes").desc())
    )
    show_and_save(task3, "task3_movies_over_2_hours")

    # 4. Актори/акторки, назви відповідних творів та персонажі.
    actor_people = (
        principals.filter(F.col("category").isin("actor", "actress"))
        .select("tconst", "nconst", "category", "characters")
    )
    title_names = (
        basics.select("tconst", F.coalesce("primaryTitle", "originalTitle").alias("work_title"))
    )
    task4 = (
        actor_people.join(names.select("nconst", "primaryName"), "nconst", "inner")
        .join(title_names, "tconst", "inner")
        .select("primaryName", "work_title", "category", "characters")
    )
    show_and_save(task4, "task4_actors_and_characters")

    # 5. Кількість дорослих фільмів/серіалів за регіонами, топ-100.
    adult_titles = (
        basics.filter(F.col("isAdult") == 1)
        .select(F.col("tconst").alias("adult_title_id"))
    )
    task5 = (
        akas.join(adult_titles, akas.titleId == adult_titles.adult_title_id, "inner")
        .filter(F.col("region").isNotNull())
        .groupBy("region")
        .agg(F.countDistinct("titleId").alias("adult_titles_count"))
        .orderBy(F.col("adult_titles_count").desc(), F.col("region").asc())
        .limit(100)
    )
    show_and_save(task5, "task5_top100_adult_titles_by_region", 100)

    # 6. Кількість епізодів у кожному TV-серіалі, топ-50.
    series = (
        basics.filter(F.col("titleType") == "tvSeries")
        .select(F.col("tconst").alias("series_id"), F.col("primaryTitle").alias("series_title"))
    )
    episode_counts = (
        episodes.groupBy(F.col("parentTconst").alias("series_id"))
        .agg(F.countDistinct("tconst").alias("episodes_count"))
    )
    task6 = (
        series.join(episode_counts, "series_id", "inner")
        .orderBy(F.col("episodes_count").desc(), F.col("series_title").asc())
        .limit(50)
    )
    show_and_save(task6, "task6_top50_series_by_episode_count", 50)

    # Спільна підготовка рейтингових даних для завдань 7 та 8.
    rated_titles = (
        basics.filter(F.col("isAdult") == 0)
        .join(ratings, "tconst", "inner")
        .filter(F.col("averageRating").isNotNull() & F.col("numVotes").isNotNull())
        .select(
            "tconst", "primaryTitle", "titleType", "startYear",
            "genres", "averageRating", "numVotes"
        )
    )

    # 7. Топ-10 найпопулярніших творів за кожне десятиліття.
    by_decade = (
        rated_titles.filter(F.col("startYear").isNotNull())
        .withColumn("decade", (F.floor(F.col("startYear") / 10) * 10).cast("int"))
    )
    decade_window = Window.partitionBy("decade").orderBy(
        F.col("averageRating").desc(), F.col("numVotes").desc()
    )
    task7 = (
        by_decade.withColumn("rank_in_decade", F.row_number().over(decade_window))
        .filter(F.col("rank_in_decade") <= 10)
        .select("decade", "primaryTitle", "titleType", "averageRating", "numVotes", "rank_in_decade")
        .orderBy(F.col("decade").desc(), F.col("rank_in_decade").asc())
    )
    show_and_save(task7, "task7_top10_by_decade", 100)

    # 8. Топ-10 найпопулярніших творів за кожним жанром.
    by_genre = (
        rated_titles.filter(F.col("genres").isNotNull() & (F.col("genres") != "\\N"))
        .withColumn("genre", F.explode(F.split(F.col("genres"), ",")))
        .filter(F.col("genre") != "")
    )
    genre_window = Window.partitionBy("genre").orderBy(
        F.col("averageRating").desc(), F.col("numVotes").desc()
    )
    task8 = (
        by_genre.withColumn("rank_in_genre", F.row_number().over(genre_window))
        .filter(F.col("rank_in_genre") <= 10)
        .select("genre", "primaryTitle", "titleType", "averageRating", "numVotes", "rank_in_genre")
        .orderBy("genre", "rank_in_genre")
    )
    show_and_save(task8, "task8_top10_by_genre", 100)

    print("\\nETL-конвеєр завершив роботу.")
    spark.stop()


if __name__ == "__main__":
    main()
