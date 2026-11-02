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

<figure class="diagram">
<div class="diagram__scroll">
<svg viewBox="0 0 720 470" style="--dg-w: 720" role="img" aria-labelledby="ch1-star-title ch1-star-desc"><title id="ch1-star-title">Star schema for retail orders</title><desc id="ch1-star-desc">A central fact_orders table holds date_id, customer_id, product_id and store_id foreign keys plus the measures quantity and amount. Each foreign key points to the primary key of one dimension table: dim_date, dim_customer, dim_product and dim_store, which hold descriptive attributes.</desc><path class="dg-line" d="M260,189 H225 V59 H190"/><path class="dg-line" d="M460,211 H495 V59 H530"/><path class="dg-line" d="M260,233 H225 V365 H190"/><path class="dg-line" d="M460,255 H495 V365 H530"/><rect class="dg-shadow" x="264" y="154" width="200" height="168"/><rect class="dg-box" x="260" y="150" width="200" height="168"/><rect class="dg-head" x="260" y="150" width="200" height="28"/><text class="dg-head-text" x="270" y="169">fact_orders</text><text class="dg-text" x="270" y="193">date_id</text><text class="dg-key" x="450" y="193" text-anchor="end">FK</text><line class="dg-rule" x1="260" y1="200" x2="460" y2="200"/><text class="dg-text" x="270" y="215">customer_id</text><text class="dg-key" x="450" y="215" text-anchor="end">FK</text><line class="dg-rule" x1="260" y1="222" x2="460" y2="222"/><text class="dg-text" x="270" y="237">product_id</text><text class="dg-key" x="450" y="237" text-anchor="end">FK</text><line class="dg-rule" x1="260" y1="244" x2="460" y2="244"/><text class="dg-text" x="270" y="259">store_id</text><text class="dg-key" x="450" y="259" text-anchor="end">FK</text><line class="dg-rule" x1="260" y1="266" x2="460" y2="266"/><text class="dg-text" x="270" y="281">quantity</text><text class="dg-sub" x="450" y="281" text-anchor="end">measure</text><line class="dg-rule" x1="260" y1="288" x2="460" y2="288"/><text class="dg-text" x="270" y="303">amount</text><text class="dg-sub" x="450" y="303" text-anchor="end">measure</text><rect class="dg-shadow" x="24" y="24" width="170" height="124"/><rect class="dg-box" x="20" y="20" width="170" height="124"/><rect class="dg-head" x="20" y="20" width="170" height="28"/><text class="dg-head-text" x="30" y="39">dim_date</text><text class="dg-text" x="30" y="63">date_id</text><text class="dg-key" x="180" y="63" text-anchor="end">PK</text><line class="dg-rule" x1="20" y1="70" x2="190" y2="70"/><text class="dg-text" x="30" y="85">date</text><line class="dg-rule" x1="20" y1="92" x2="190" y2="92"/><text class="dg-text" x="30" y="107">week_start</text><line class="dg-rule" x1="20" y1="114" x2="190" y2="114"/><text class="dg-text" x="30" y="129">month</text><rect class="dg-shadow" x="534" y="24" width="170" height="124"/><rect class="dg-box" x="530" y="20" width="170" height="124"/><rect class="dg-head" x="530" y="20" width="170" height="28"/><text class="dg-head-text" x="540" y="39">dim_customer</text><text class="dg-text" x="540" y="63">customer_id</text><text class="dg-key" x="690" y="63" text-anchor="end">PK</text><line class="dg-rule" x1="530" y1="70" x2="700" y2="70"/><text class="dg-text" x="540" y="85">name</text><line class="dg-rule" x1="530" y1="92" x2="700" y2="92"/><text class="dg-text" x="540" y="107">segment</text><line class="dg-rule" x1="530" y1="114" x2="700" y2="114"/><text class="dg-text" x="540" y="129">city</text><rect class="dg-shadow" x="24" y="330" width="170" height="124"/><rect class="dg-box" x="20" y="326" width="170" height="124"/><rect class="dg-head" x="20" y="326" width="170" height="28"/><text class="dg-head-text" x="30" y="345">dim_product</text><text class="dg-text" x="30" y="369">product_id</text><text class="dg-key" x="180" y="369" text-anchor="end">PK</text><line class="dg-rule" x1="20" y1="376" x2="190" y2="376"/><text class="dg-text" x="30" y="391">name</text><line class="dg-rule" x1="20" y1="398" x2="190" y2="398"/><text class="dg-text" x="30" y="413">category</text><line class="dg-rule" x1="20" y1="420" x2="190" y2="420"/><text class="dg-text" x="30" y="435">brand</text><rect class="dg-shadow" x="534" y="330" width="170" height="124"/><rect class="dg-box" x="530" y="326" width="170" height="124"/><rect class="dg-head" x="530" y="326" width="170" height="28"/><text class="dg-head-text" x="540" y="345">dim_store</text><text class="dg-text" x="540" y="369">store_id</text><text class="dg-key" x="690" y="369" text-anchor="end">PK</text><line class="dg-rule" x1="530" y1="376" x2="700" y2="376"/><text class="dg-text" x="540" y="391">store_name</text><line class="dg-rule" x1="530" y1="398" x2="700" y2="398"/><text class="dg-text" x="540" y="413">city</text><line class="dg-rule" x1="530" y1="420" x2="700" y2="420"/><text class="dg-text" x="540" y="435">region</text><circle class="dg-dot" cx="190" cy="59" r="3"/><circle class="dg-dot" cx="530" cy="59" r="3"/><circle class="dg-dot" cx="190" cy="365" r="3"/><circle class="dg-dot" cx="530" cy="365" r="3"/><text class="dg-sub" x="360" y="352" text-anchor="middle">grain: one row per order line</text><text class="dg-sub" x="360" y="370" text-anchor="middle">FK → PK: every dimension</text><text class="dg-sub" x="360" y="388" text-anchor="middle">is one join away</text></svg>
</div>
<figcaption>Star schema for retail orders: one fact table at the centre, four dimensions around it.</figcaption>
</figure>

A query joins the fact table to the dimensions it needs, filters on dimension attributes ("segment = 'premium'") and aggregates the measures ("SUM(amount)"). Every dimension is one join away, so queries stay simple and fast.

**Choose the grain first.** The finest grain (one row per order line) answers every question but costs the most storage. A coarser grain (daily totals per store) is cheaper but can never be broken down again. Declare the grain before adding measures, and keep every measure true at that grain.

**Three kinds of fact table:**

| Type                  | One row per                                | Example                                                       |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| Transaction           | Event                                      | Each order line, each click                                   |
| Periodic snapshot     | Entity per period                          | Each account's balance at the end of each day                 |
| Accumulating snapshot | Process instance, updated as it progresses | Each order, with timestamps for placed, shipped and delivered |

**Normalization and snowflake schemas.** **Normalization** splits data into separate tables so each fact is stored once; **denormalization** repeats data on purpose to avoid joins. A star schema keeps its dimensions denormalized: the category name is repeated on every product row. A **snowflake schema** normalizes the dimensions into sub-tables (product → category → department), so the schema branches out like a snowflake.

<figure class="diagram">
<div class="diagram__scroll">
<svg viewBox="0 0 740 340" style="--dg-w: 740" role="img" aria-labelledby="ch1-snow-title ch1-snow-desc"><title id="ch1-snow-title">Snowflake schema for retail orders</title><desc id="ch1-snow-desc">The fact_orders table links to dim_date and dim_customer as in a star schema, but the product dimension is normalized into dim_product, dim_category and dim_department, and the store dimension into dim_store, dim_city and dim_region, each linked by a foreign key to the next.</desc><path class="dg-line" d="M560,151 H547 V59 H534"/><path class="dg-line" d="M560,173 H547 V249 H534"/><path class="dg-line" d="M374,103 H363 V59 H352"/><path class="dg-line" d="M192,103 H181 V59 H170"/><path class="dg-line" d="M374,293 H363 V249 H352"/><path class="dg-line" d="M192,293 H181 V249 H170"/><path class="dg-line" d="M645,90 V48"/><path class="dg-line" d="M645,258 V290"/><rect class="dg-shadow" x="564" y="94" width="170" height="168"/><rect class="dg-box" x="560" y="90" width="170" height="168"/><rect class="dg-head" x="560" y="90" width="170" height="28"/><text class="dg-head-text" x="570" y="109">fact_orders</text><text class="dg-text" x="570" y="133">date_id</text><text class="dg-key" x="720" y="133" text-anchor="end">FK</text><line class="dg-rule" x1="560" y1="140" x2="730" y2="140"/><text class="dg-text" x="570" y="155">product_id</text><text class="dg-key" x="720" y="155" text-anchor="end">FK</text><line class="dg-rule" x1="560" y1="162" x2="730" y2="162"/><text class="dg-text" x="570" y="177">store_id</text><text class="dg-key" x="720" y="177" text-anchor="end">FK</text><line class="dg-rule" x1="560" y1="184" x2="730" y2="184"/><text class="dg-text" x="570" y="199">customer_id</text><text class="dg-key" x="720" y="199" text-anchor="end">FK</text><line class="dg-rule" x1="560" y1="206" x2="730" y2="206"/><text class="dg-text" x="570" y="221">quantity</text><text class="dg-sub" x="720" y="221" text-anchor="end">measure</text><line class="dg-rule" x1="560" y1="228" x2="730" y2="228"/><text class="dg-text" x="570" y="243">amount</text><text class="dg-sub" x="720" y="243" text-anchor="end">measure</text><rect class="dg-shadow" x="564" y="24" width="170" height="28"/><rect class="dg-box" x="560" y="20" width="170" height="28"/><rect class="dg-head" x="560" y="20" width="170" height="28"/><text class="dg-head-text" x="570" y="39">dim_date</text><rect class="dg-shadow" x="564" y="294" width="170" height="28"/><rect class="dg-box" x="560" y="290" width="170" height="28"/><rect class="dg-head" x="560" y="290" width="170" height="28"/><text class="dg-head-text" x="570" y="309">dim_customer</text><rect class="dg-shadow" x="378" y="24" width="160" height="102"/><rect class="dg-box" x="374" y="20" width="160" height="102"/><rect class="dg-head" x="374" y="20" width="160" height="28"/><text class="dg-head-text" x="384" y="39">dim_product</text><text class="dg-text" x="384" y="63">product_id</text><text class="dg-key" x="524" y="63" text-anchor="end">PK</text><line class="dg-rule" x1="374" y1="70" x2="534" y2="70"/><text class="dg-text" x="384" y="85">name</text><line class="dg-rule" x1="374" y1="92" x2="534" y2="92"/><text class="dg-text" x="384" y="107">category_id</text><text class="dg-key" x="524" y="107" text-anchor="end">FK</text><rect class="dg-shadow" x="196" y="24" width="160" height="102"/><rect class="dg-box" x="192" y="20" width="160" height="102"/><rect class="dg-head" x="192" y="20" width="160" height="28"/><text class="dg-head-text" x="202" y="39">dim_category</text><text class="dg-text" x="202" y="63">category_id</text><text class="dg-key" x="342" y="63" text-anchor="end">PK</text><line class="dg-rule" x1="192" y1="70" x2="352" y2="70"/><text class="dg-text" x="202" y="85">category_name</text><line class="dg-rule" x1="192" y1="92" x2="352" y2="92"/><text class="dg-text" x="202" y="107">department_id</text><text class="dg-key" x="342" y="107" text-anchor="end">FK</text><rect class="dg-shadow" x="14" y="24" width="160" height="80"/><rect class="dg-box" x="10" y="20" width="160" height="80"/><rect class="dg-head" x="10" y="20" width="160" height="28"/><text class="dg-head-text" x="20" y="39">dim_department</text><text class="dg-text" x="20" y="63">department_id</text><text class="dg-key" x="160" y="63" text-anchor="end">PK</text><line class="dg-rule" x1="10" y1="70" x2="170" y2="70"/><text class="dg-text" x="20" y="85">department_name</text><rect class="dg-shadow" x="378" y="214" width="160" height="102"/><rect class="dg-box" x="374" y="210" width="160" height="102"/><rect class="dg-head" x="374" y="210" width="160" height="28"/><text class="dg-head-text" x="384" y="229">dim_store</text><text class="dg-text" x="384" y="253">store_id</text><text class="dg-key" x="524" y="253" text-anchor="end">PK</text><line class="dg-rule" x1="374" y1="260" x2="534" y2="260"/><text class="dg-text" x="384" y="275">store_name</text><line class="dg-rule" x1="374" y1="282" x2="534" y2="282"/><text class="dg-text" x="384" y="297">city_id</text><text class="dg-key" x="524" y="297" text-anchor="end">FK</text><rect class="dg-shadow" x="196" y="214" width="160" height="102"/><rect class="dg-box" x="192" y="210" width="160" height="102"/><rect class="dg-head" x="192" y="210" width="160" height="28"/><text class="dg-head-text" x="202" y="229">dim_city</text><text class="dg-text" x="202" y="253">city_id</text><text class="dg-key" x="342" y="253" text-anchor="end">PK</text><line class="dg-rule" x1="192" y1="260" x2="352" y2="260"/><text class="dg-text" x="202" y="275">city_name</text><line class="dg-rule" x1="192" y1="282" x2="352" y2="282"/><text class="dg-text" x="202" y="297">region_id</text><text class="dg-key" x="342" y="297" text-anchor="end">FK</text><rect class="dg-shadow" x="14" y="214" width="160" height="80"/><rect class="dg-box" x="10" y="210" width="160" height="80"/><rect class="dg-head" x="10" y="210" width="160" height="28"/><text class="dg-head-text" x="20" y="229">dim_region</text><text class="dg-text" x="20" y="253">region_id</text><text class="dg-key" x="160" y="253" text-anchor="end">PK</text><line class="dg-rule" x1="10" y1="260" x2="170" y2="260"/><text class="dg-text" x="20" y="275">region_name</text><circle class="dg-dot" cx="534" cy="59" r="3"/><circle class="dg-dot" cx="534" cy="249" r="3"/><circle class="dg-dot" cx="352" cy="59" r="3"/><circle class="dg-dot" cx="170" cy="59" r="3"/><circle class="dg-dot" cx="352" cy="249" r="3"/><circle class="dg-dot" cx="170" cy="249" r="3"/><text class="dg-sub" x="10" y="150">product hierarchy: 3 normalized tables</text><text class="dg-sub" x="10" y="198">store hierarchy: 3 normalized tables</text></svg>
</div>
<figcaption>Snowflake schema: the product and store dimensions are normalized into chains of sub-tables, so a query needs more joins.</figcaption>
</figure>

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

<figure class="diagram">
<div class="diagram__scroll">
<svg viewBox="0 0 810 270" style="--dg-w: 810" role="img" aria-labelledby="ch1-flow-title ch1-flow-desc"><title id="ch1-flow-title">Data flow for an ML system</title><desc id="ch1-flow-desc">App databases, event logs, third-party APIs and media files flow into a data lake of raw files. SQL turns lake data into warehouse tables and encoders turn media into embeddings. Both feed a feature store, whose offline store serves training and whose online store serves predictions.</desc><defs><marker id="ch1-flow-arrow" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto"><path class="dg-arrowhead" d="M0,0 L8,4 L0,8 z"/></marker></defs><text class="dg-sub" x="80" y="22" text-anchor="middle">sources</text><text class="dg-sub" x="240" y="22" text-anchor="middle">storage</text><text class="dg-sub" x="405" y="22" text-anchor="middle">processed</text><text class="dg-sub" x="575" y="22" text-anchor="middle">features</text><text class="dg-sub" x="740" y="22" text-anchor="middle">models</text><rect class="dg-shadow" x="8" y="44" width="150" height="40"/><rect class="dg-box" x="4" y="40" width="150" height="40"/><text class="dg-strong" x="79" y="65" text-anchor="middle">app databases</text><path class="dg-line" d="M154,60 H165"/><rect class="dg-shadow" x="8" y="99" width="150" height="40"/><rect class="dg-box" x="4" y="95" width="150" height="40"/><text class="dg-strong" x="79" y="120" text-anchor="middle">event logs</text><path class="dg-line" d="M154,115 H165"/><rect class="dg-shadow" x="8" y="154" width="150" height="40"/><rect class="dg-box" x="4" y="150" width="150" height="40"/><text class="dg-strong" x="79" y="175" text-anchor="middle">third-party APIs</text><path class="dg-line" d="M154,170 H165"/><rect class="dg-shadow" x="8" y="209" width="150" height="40"/><rect class="dg-box" x="4" y="205" width="150" height="40"/><text class="dg-strong" x="79" y="230" text-anchor="middle">media files</text><path class="dg-line" d="M154,225 H165"/><path class="dg-line" d="M165,60 V225"/><path class="dg-line" d="M165,145 H180" marker-end="url(#ch1-flow-arrow)"/><rect class="dg-shadow" x="184" y="99" width="120" height="100"/><rect class="dg-box" x="180" y="95" width="120" height="100"/><text class="dg-strong" x="240" y="143" text-anchor="middle">data lake</text><text class="dg-sub" x="240" y="160" text-anchor="middle">raw files</text><path class="dg-line" d="M300,125 H320 V85 H340" marker-end="url(#ch1-flow-arrow)"/><path class="dg-line" d="M300,165 H320 V205 H340" marker-end="url(#ch1-flow-arrow)"/><rect class="dg-shadow" x="344" y="64" width="130" height="50"/><rect class="dg-box" x="340" y="60" width="130" height="50"/><text class="dg-strong" x="405" y="83" text-anchor="middle">warehouse</text><text class="dg-sub" x="405" y="100" text-anchor="middle">tables, via SQL</text><rect class="dg-shadow" x="344" y="184" width="130" height="50"/><rect class="dg-box" x="340" y="180" width="130" height="50"/><text class="dg-strong" x="405" y="203" text-anchor="middle">embeddings</text><text class="dg-sub" x="405" y="220" text-anchor="middle">via encoders</text><path class="dg-line" d="M470,85 H490 V130 H510" marker-end="url(#ch1-flow-arrow)"/><path class="dg-line" d="M470,205 H490 V160 H510" marker-end="url(#ch1-flow-arrow)"/><rect class="dg-shadow" x="514" y="114" width="130" height="70"/><rect class="dg-box" x="510" y="110" width="130" height="70"/><text class="dg-strong" x="575" y="143" text-anchor="middle">feature store</text><text class="dg-sub" x="575" y="160" text-anchor="middle">one definition</text><path class="dg-line" d="M640,135 H660 V105 H680" marker-end="url(#ch1-flow-arrow)"/><path class="dg-line" d="M640,155 H660 V185 H680" marker-end="url(#ch1-flow-arrow)"/><rect class="dg-shadow" x="684" y="84" width="120" height="50"/><rect class="dg-box" x="680" y="80" width="120" height="50"/><text class="dg-strong" x="740" y="103" text-anchor="middle">training</text><text class="dg-sub" x="740" y="120" text-anchor="middle">offline store</text><rect class="dg-shadow" x="684" y="164" width="120" height="50"/><rect class="dg-box" x="680" y="160" width="120" height="50"/><text class="dg-strong" x="740" y="183" text-anchor="middle">serving</text><text class="dg-sub" x="740" y="200" text-anchor="middle">online store</text></svg>
</div>
<figcaption>Data flow for an ML system: sources land in the lake, become tables and embeddings, and reach training and serving through one feature store.</figcaption>
</figure>

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
