---
title: "Data Foundations"
chapter: 1
permalink: "/blog/cracking-the-ml-interview/data-foundations/"
updated: 2026-11-01
description: "Every ML system begins with a question an interviewer will ask in the first five minutes: where does the data come from, what shape is it in, and how does it reach the model?"
---

Every ML system begins with a question an interviewer will ask in the first five minutes: where does the data come from, what shape is it in, and how does it reach the model? This chapter gives you the vocabulary to answer that crisply: warehouses, lakes and lakehouses, and the difference between structured, semi-structured and unstructured data.

## 1.1 Intuition

**Two kinds of databases, two kinds of work.** A company's app runs on an _operational_ database that records each order, login or payment as it happens, one row at a time, and must answer in milliseconds. This workload is **OLTP (Online Transaction Processing)**: many small, fast reads and writes that keep a live application running. A **data warehouse** is a separate store built for **OLAP (Online Analytical Processing)**: queries that scan and aggregate large amounts of historical data, such as "average basket size by city, per week, for the last two years".

Think of OLTP as a shop's till and the warehouse as its accounting office. The till must be fast for one customer at a time; the office reconciles everything, keeps history, and answers questions across months. You would never run a year-long report on the till while customers wait.

|                | OLTP (operational database) | OLAP (data warehouse)                   |
| -------------- | --------------------------- | --------------------------------------- |
| Typical query  | Read or write one record    | Aggregate over millions of rows         |
| Users          | The application, end users  | Analysts, data scientists, ML pipelines |
| Data           | Current state, normalized   | Cleaned history, often denormalized     |
| Storage layout | Row-oriented                | Column-oriented                         |
| Examples       | PostgreSQL, MySQL           | BigQuery, Snowflake, Redshift           |

**How data gets into the warehouse.** A pipeline copies data from many sources: app databases, event logs, and third-party **APIs (Application Programming Interfaces)**, the defined ways one program requests data from another. In **ETL** (extract, transform, load) the data is cleaned before loading. In **ELT** (extract, load, transform) raw data is loaded first and transformed inside the warehouse with **SQL (Structured Query Language)**, the standard language for querying tables. Modern cloud warehouses make that cheap, and ELT keeps the raw copy, so you can re-derive tables when a bug is found.

**Lakes and lakehouses.** A **data lake** is cheap object storage (Amazon S3, Google Cloud Storage) that holds everything in its raw form: logs, JSON files, images, audio. **JSON (JavaScript Object Notation)** is a text format of nested key-value pairs and the usual format for app events. A lake uses _schema-on-read_: structure is applied when you query, not when you store. A warehouse uses _schema-on-write_: data must fit the table's schema to get in. A **lakehouse** adds warehouse features (reliable transactions, schemas, fast SQL) on top of lake storage through table formats such as Delta Lake and Apache Iceberg.

|             | Data warehouse                                                         | Data lake                         | Lakehouse                              |
| ----------- | ---------------------------------------------------------------------- | --------------------------------- | -------------------------------------- |
| Holds       | Cleaned, structured tables                                             | Anything, raw                     | Anything, with managed tables on top   |
| Schema      | On write                                                               | On read                           | On write for tables, raw files allowed |
| Cost per TB | Higher                                                                 | Lowest                            | Low                                    |
| Best for    | Business intelligence (BI) dashboards, SQL analytics, tabular features | Raw archives, images, text, audio | One copy of data for BI and ML         |

**Star schemas.** Warehouses usually organize tables as a **star schema**, named for its shape: one central fact table with dimension tables around it like the points of a star. It is the most common warehouse design, and interviewers use it to test whether you can model data, not just query it.

- A **fact table** records business events, one row per event at a fixed **grain** (the level of detail one row represents), for example one row per order line. It holds numeric **measures** (amount, quantity) and a **foreign key (FK)** to each dimension: a column that points to a row in another table. Fact tables are long and narrow: billions of rows, few columns.
- A **dimension table** describes the who, what, where and when of each event: customers, products, stores, dates. It holds descriptive **attributes** (name, category, city) and a **primary key (PK)**, the column that uniquely identifies each row. That key is usually a **surrogate key**, a meaningless integer the warehouse assigns, so it stays stable even if the business ID changes. Dimension tables are short and wide.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>star schema · retail orders, 1 fact table, 4 dimensions</figcaption></figure>

A query joins the fact table to the dimensions it needs, filters on dimension attributes ("segment = 'premium'") and aggregates the measures ("SUM(amount)"). Every dimension is one join away, so queries stay simple and fast.

**Choose the grain first.** The finest grain (one row per order line) answers every question but costs the most storage. A coarser grain (daily totals per store) is cheaper but can never be broken down again. Declare the grain before adding measures, and keep every measure true at that grain.

**Three kinds of fact table:**

| Type                  | One row per                                | Example                                                       |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| Transaction           | Event                                      | Each order line, each click                                   |
| Periodic snapshot     | Entity per period                          | Each account's balance at the end of each day                 |
| Accumulating snapshot | Process instance, updated as it progresses | Each order, with timestamps for placed, shipped and delivered |

**Normalization and snowflake schemas.** **Normalization** splits data into separate tables so each fact is stored once; **denormalization** repeats data on purpose to avoid joins. A star schema keeps its dimensions denormalized: the category name is repeated on every product row. A **snowflake schema** normalizes the dimensions into sub-tables (product → category → department), so the schema branches out like a snowflake.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>snowflake schema · two dimensions normalized into sub-tables</figcaption></figure>

|                            | Star schema                     | Snowflake schema                                        |
| -------------------------- | ------------------------------- | ------------------------------------------------------- |
| Dimension tables           | Denormalized, one per dimension | Normalized into sub-tables                              |
| Joins per query            | Fewer                           | More                                                    |
| Storage                    | More duplication                | Less duplication                                        |
| Query speed and simplicity | Faster, simpler                 | Slower, more complex                                    |
| Typical use                | Most analytics warehouses       | Very large dimensions, or hierarchies that change often |

**Slowly changing dimensions (SCDs).** Dimension attributes change over time: a customer moves city, a product changes category. An SCD strategy decides what happens to history. **Type 1** overwrites the old value, so history is lost. **Type 2** adds a new row with `valid_from` and `valid_to` dates and keeps the old one. Type 2 matters for ML: to build features as they were at prediction time, join each fact to the dimension row that was valid at that moment.

**Structured, semi-structured, unstructured.**

| Type            | What it is                       | Examples                                                                      | Where it lives                                                       | How a model consumes it                                             |
| --------------- | -------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Structured      | Fixed columns and types          | Transactions, user profiles, sensor readings                                  | Relational DB, warehouse                                             | Directly as features                                                |
| Semi-structured | Self-describing, flexible fields | JSON events, XML (tagged text, like HTML for data), logs, emails with headers | Lake, NoSQL (non-relational) databases such as MongoDB, JSON columns | Flatten or parse, then featurize                                    |
| Unstructured    | No predefined data model         | Images, video, audio, free text, PDFs                                         | Object storage (lake)                                                | Pretrained encoders turn it into embeddings (dense numeric vectors) |

Most real-world data is unstructured. The usual pattern stores the raw file in object storage, keeps its metadata (ID, path, label, timestamp) in a structured table, and stores learned embeddings for search or features.

## 1.2 Core math: the numbers that matter

This topic has little formal math, but interviewers expect two quantitative habits.

**Back-of-the-envelope sizing.** Estimate volume before you choose storage or a pipeline:

$$
\text{daily volume} = \text{users} \times \text{events per user per day} \times \text{bytes per event}
$$

For 100 million users, 50 events each per day and 200 bytes per event: 10^8 × 50 × 200 = 10^12 bytes, about 1 TB per day, or roughly 365 TB per year. That number tells you raw events belong in a lake and that training jobs should read aggregated or sampled tables.

**Why columnar storage wins for analytics.** A row store reads whole rows; a column store reads only the columns the query touches. If a table has m columns of similar width and a query needs k of them, a column store reads roughly:

$$
\frac{\text{bytes read (columnar)}}{\text{bytes read (row)}} \approx \frac{k}{m}
$$

A query using 3 of 100 columns reads about 3% of the data, before compression. Columns hold values of one type, so they also compress far better.

**Point-in-time correctness.** A feature used to predict a label at time t may use only data recorded strictly before t:

$$
x(t) = f\left(\{\, e_i : \text{timestamp}(e_i) < t \,\}\right)
$$

Breaking this rule is called **leakage**: the model sees the future in training and fails in production. It is the most common data bug in ML interviews and in real systems.

## 1.3 Code

**A star-schema query that builds a training table.** Weekly spend per customer segment, joining a fact table to two dimensions:

```sql
SELECT
  d.week_start,
  c.segment,
  SUM(f.amount)               AS total_spend,
  COUNT(DISTINCT f.customer_id) AS active_customers
FROM fact_orders f
JOIN dim_customer c ON f.customer_id = c.customer_id
JOIN dim_date     d ON f.date_id     = d.date_id
WHERE d.week_start >= DATE '2025-01-01'
GROUP BY d.week_start, c.segment;
```

**Flattening semi-structured events.** Nested JSON becomes dotted columns; missing fields become NaN:

```python
import pandas as pd

events = [
    {"user": {"id": 7, "country": "IN"}, "event": "click", "props": {"item_id": 42, "price": 299.0}},
    {"user": {"id": 9, "country": "AE"}, "event": "view",  "props": {"item_id": 17}},
]
flat = pd.json_normalize(events)
print(flat.columns.tolist())
# ['event', 'user.id', 'user.country', 'props.item_id', 'props.price']
```

**Reading only the columns you need from a columnar file:**

```python
df = pd.read_parquet("events.parquet", columns=["user_id", "ts", "price"])
```

**A point-in-time join that prevents leakage.** Each label gets the latest feature value recorded strictly before the label's timestamp:

```python
labels = pd.DataFrame({"user_id": [1, 1],
                       "ts": pd.to_datetime(["2026-01-10", "2026-01-20"]),
                       "label": [0, 1]})
feats = pd.DataFrame({"user_id": [1, 1, 1],
                      "ts": pd.to_datetime(["2026-01-05", "2026-01-15", "2026-01-25"]),
                      "spend_30d": [100, 180, 250]})

train = pd.merge_asof(labels.sort_values("ts"), feats.sort_values("ts"),
                      on="ts", by="user_id",
                      direction="backward", allow_exact_matches=False)
# label on Jan 10 gets spend_30d=100; label on Jan 20 gets 180, never 250
```

A naive join on `user_id` that takes the latest feature would attach 250 to both rows: a feature from the future.

## 1.4 In system design

In a design round, open the data section by naming four things: the **sources**, their **structure**, the **volume and freshness** needed, and where the **labels** come from. A typical answer traces data from sources through a lake and a warehouse to training, as in the diagram below.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>data flow for an ML system · sources to serving</figcaption></figure>

Raw files land in the lake. SQL turns them into warehouse tables, encoders turn media into embeddings, and the feature store serves the same features to training and to serving.

- **Tabular features** (transactions, profiles, aggregates) come from warehouse tables or a **feature store**, which keeps an offline copy for training and a low-latency online copy for serving, so both compute features the same way.
- **Unstructured inputs** (product images, reviews) live in object storage. Pipelines encode them into embeddings once and store the vectors, rather than re-encoding raw files on every training run.
- **Freshness** decides batch versus streaming. Daily recommendations can use nightly batch jobs; fraud detection needs features updated within seconds. That calls for streaming tools such as Apache Kafka, which moves events between systems, and Apache Flink, which computes features on them as they arrive.
- **Labels** often arrive late: a chargeback can take weeks to confirm fraud. Say how long the delay is and how you join labels back to the features as they were at prediction time.

## 1.5 Interview Q&A

**Q1. What is a data warehouse, and why not train models straight off the production database?** A warehouse is a central store of cleaned, historical data organized for analytical queries. The production database is tuned for many small reads and writes; heavy scans would slow the live app, it usually keeps only current state rather than history, and its data is scattered across normalized tables. The warehouse fixes all three.

**Q2. Data lake versus data warehouse: when would you use each?** Use a lake for cheap storage of raw and unstructured data in any format, with structure applied when read. Use a warehouse for curated, structured tables that need fast SQL and consistent definitions. Many teams keep both, or use a lakehouse to get warehouse-style tables over lake storage.

**Q3. ETL or ELT?** ETL transforms data before loading, which suits strict schemas or sensitive data that must be cleaned or masked first. ELT loads raw data and transforms it inside the warehouse; it is the modern default because compute is cheap and the raw copy lets you fix a transformation bug and rebuild.

**Q4. Explain a star schema.** A central fact table holds events and numeric measures with foreign keys; surrounding dimension tables hold descriptive attributes. Analytics queries join the fact table to a few dimensions and aggregate. Follow-up: a snowflake schema normalizes the dimensions, saving space at the cost of more joins.

**Q5. Why are column stores faster for analytics?** Analytical queries touch few columns over many rows. A column store reads only those columns, compresses well because each column has one type, and supports vectorized execution. Row stores win for transactional work that reads or writes whole records.

**Q6. Give examples of structured, semi-structured and unstructured data, and say how each enters a model.** Structured: a payments table, used directly as features. Semi-structured: JSON clickstream events, flattened and aggregated. Unstructured: images or support tickets, passed through a pretrained encoder to get embeddings: a **CNN (convolutional neural network)** or vision transformer for images, a text encoder for tickets.

**Q7. You are building an image classifier for 50 million product photos. How do you store the data?** Keep image files in object storage, partitioned sensibly (by date or category). Keep a structured table with image ID, storage path, label, label source and split assignment. Precompute embeddings if several models reuse them, and version the dataset so each model can be traced to the exact data it saw.

**Q8. What is point-in-time correctness, and how do you enforce it?** Features for a label at time t must be built only from data recorded before t. Enforce it with timestamped feature tables, as-of joins, and time-based train/test splits. A feature store with point-in-time joins does this for you.

**Q9. When do you need streaming data rather than batch?** When the value of a prediction decays within minutes: fraud, ride pricing, feed ranking after a user's last few clicks. Streaming adds operational cost, so justify it with the freshness requirement; otherwise prefer batch.

**Q10. What is the grain of a fact table, and how do you choose it?** The grain is what one row represents, such as one order line or one store-day. Choose the finest grain the questions and budget allow, because you can always aggregate up but never break a coarse row back down. State the grain before listing measures, and keep every measure true at that grain.

**Q11. What is a slowly changing dimension, and why does Type 2 matter for ML?** It is a dimension whose attributes change over time, like a customer's city. Type 1 overwrites the value and loses history; Type 2 adds a new row with validity dates. For training data you need the attribute as it was when the event happened, which only Type 2 can give; joining to the current Type 1 value leaks later information into old rows.

## 1.6 Cheat sheet and pitfalls

**Remember:**

- OLTP serves the app (row store, current state); OLAP serves analysis (column store, history).
- Warehouse = schema-on-write, curated. Lake = schema-on-read, raw. Lakehouse = lake storage plus warehouse tables.
- ELT is the modern default; it keeps the raw copy.
- Star schema: one fact table (events, measures) plus dimension tables (who, what, where, when).
- Unstructured data is stored as files, described by structured metadata, and fed to models as embeddings.
- Sizing: users × events × bytes, then decide lake versus warehouse and batch versus stream.

**Pitfalls that cost candidates:**

- Skipping the data section in a design round and jumping straight to the model.
- Proposing to train on the production database.
- Joining the latest feature values to old labels, which leaks the future.
- Calling JSON "unstructured"; it is semi-structured.
- Storing raw images inside warehouse rows instead of storing paths to object storage.
- Ignoring label delay, so the most recent training rows carry wrong labels.
- Choosing streaming because it sounds advanced, without a freshness requirement that needs it.

## 1.7 Further reading

1. [What is a data warehouse?](https://cloud.google.com/learn/what-is-a-data-warehouse) (Google Cloud)
2. [Structured vs. unstructured data](https://www.ibm.com/blog/structured-vs-unstructured-data/) (IBM)
