# Supabase Migration Reference — MedicareCostAnalysis

**Inventory date:** 2026-05-11
**Purpose:** Reference document to evaluate moving off Supabase to a local Postgres, Neon, Railway Postgres, or other cheaper option.

---

## 1. Project metadata

| Field | Value |
|---|---|
| Project name | **MedicareCostAnalysis** |
| Project ID / ref | `zdkoniqnvbklxtsviikl` |
| API URL | `https://zdkoniqnvbklxtsviikl.supabase.co` |
| DB host | `db.zdkoniqnvbklxtsviikl.supabase.co` |
| Region | `us-east-1` (AWS) |
| Postgres version | **17.6.1.104** (ga channel) |
| Organization | `psalrjomyadtwlzcaaxu` |
| Created | 2026-04-08 |
| Status | `ACTIVE_HEALTHY` |

---

## 2. Database size and table breakdown

**Total DB size: 29 MB**, of which **18 MB is `public` schema**. Tiny — fits comfortably in any free tier.

| Table | Rows | Total size | Heap | Indexes+TOAST |
|---|---:|---:|---:|---:|
| `lstm_forecasts` | 62,703 | **9.7 MB** | 4.7 MB | 5 MB |
| `stage1_allowed_amounts` | 33,318 | 5 MB | 2.2 MB | 2.9 MB |
| `stage2_oop_estimates` | 23,424 | 3.2 MB | 1.6 MB | 1.6 MB |
| `specialty_yearly_avg` | 1,067 | 120 kB | 48 kB | 72 kB |
| `lookup_labels` | 204 | 96 kB | 16 kB | 80 kB |
| `model_metrics` | 20 | 48 kB | 8 kB | 40 kB |
| `feature_importances` | 10 | 48 kB | 8 kB | 40 kB |
| `state_summary` | 63 | 48 kB | 8 kB | 40 kB |
| **Total (public)** | **~120k rows** | **18 MB** | | |

**Observations:**
- This is a **read-only model output store**, not a transactional database. Forecasts and stage outputs are computed offline (LSTM + ML pipeline per the project's `MODELING.md` and `V2_MODEL_SPEC.md`) and bulk-inserted.
- Sizes are tiny; any free Postgres tier handles this.
- `specialty_idx` / `hcpcs_bucket` / `state_idx` columns are `smallint` — disciplined schema, low memory footprint.

---

## 3. Schema inventory

| Metric | Count |
|---|---:|
| Tables (public schema) | **8** |
| Indexes | **21** |
| Foreign keys | **0** (denormalized model output, joins via `*_idx` columns) |
| Views | 0 |
| Functions | 1 (`public.exec_sql` — **security issue**, see §6) |
| Materialized views | 0 |

### Tables (8) and column shapes

| Table | Schema highlights |
|---|---|
| `lookup_labels` | `(id, category text, idx int, label text)` — joins back to `*_idx` columns in other tables to denormalize labels |
| `stage1_allowed_amounts` | `(specialty_idx, hcpcs_bucket, state_idx, place_of_service, n_records, mean_allowed, median_allowed, p10/p90, mean_charge, mean_risk_score)` — Stage 1 model output |
| `stage2_oop_estimates` | `(specialty_idx, hcpcs_bucket, census_region, dual_eligible, has_supplemental, age_group, income_bracket, n_records, oop_p10/p50/p90, mean_allowed)` — Stage 2 OOP estimates |
| `lstm_forecasts` | `(specialty_idx, hcpcs_bucket, state_idx, forecast_year, forecast_mean/std/p10/p50/p90, last_known_year, last_known_value, n_history_years)` — 62k LSTM forecasts |
| `state_summary` | `(state_abbrev unique, state_idx, mean_allowed, median_allowed, n_records)` |
| `model_metrics` | `(model_name, stage, metric_name, metric_value)` |
| `feature_importances` | `(model_name, feature_name, importance, rank)` |
| `specialty_yearly_avg` | `(specialty_idx, year, mean_allowed) PK(specialty_idx, year)` |

### Schema management

7 migrations in Supabase's `supabase_migrations.schema_migrations`:

| Version | Name |
|---|---|
| 20260408225148 | `create_lookup_labels` |
| 20260408225158 | `create_stage1_allowed_amounts` |
| 20260408225208 | `create_stage2_oop_estimates` |
| 20260408225217 | `create_lstm_forecasts` |
| 20260408225230 | `create_state_summary` |
| 20260408225232 | `create_model_metrics` |
| 20260408225233 | `create_feature_importances` |

**`specialty_yearly_avg` is NOT in the migration history** — it was created out-of-band via SQL editor or direct connection. Pre-migration, dump it and add it to the migration sequence to keep the schema reproducible.

Migration history can be exported via Supabase's CLI:
```bash
supabase db pull --project-ref zdkoniqnvbklxtsviikl
```
This produces a flat `supabase/migrations/*.sql` directory you can carry to any other Postgres.

---

## 4. Postgres extensions installed (non-default)

Only these 5 extensions are actually installed:

| Extension | Schema | Version | Purpose |
|---|---|---|---|
| `plpgsql` | `pg_catalog` | 1.0 | Default. |
| `pg_stat_statements` | `extensions` | 1.11 | Query statistics for the Supabase advisor. |
| `pgcrypto` | `extensions` | 1.3 | Crypto functions. Verify if used by the app. |
| `uuid-ossp` | `extensions` | 1.1 | UUID generation. Verify if used. |
| `supabase_vault` | `vault` | 0.3.1 | Supabase-only encrypted secrets. **Likely unused.** |

**Migration implication:** zero portability concerns. Vanilla Postgres 17 hosts this perfectly. Verify Vault is empty before migrating:
```sql
SELECT count(*) FROM vault.secrets;
```

---

## 5. Edge functions, Realtime, Auth, Storage

| Surface | State | Migration impact |
|---|---|---|
| **Edge functions** | **0 deployed** | None. |
| **Realtime publication** | `supabase_realtime` (default) | Verify frontend doesn't subscribe; drop if not. |
| **Auth users** | **0** | No user migration needed. |
| **Storage objects** | **0** | No buckets. |
| **Vault secrets** | likely 0 (verify) | — |

This is **Postgres-only** in practice. Migration target evaluation is purely "what's the cheapest place to run a 29 MB Postgres database."

---

## 6. Security posture

### `public.exec_sql` — intentional dev-time tool (decision: keep on Supabase)

The security advisor flags this at ERROR level:

> Function `public.exec_sql(query text)` can be executed by the `anon` role as a `SECURITY DEFINER` function via `/rest/v1/rpc/exec_sql`.
>
> Function `public.exec_sql(query text)` can be executed by the `authenticated` role as a `SECURITY DEFINER` function via `/rest/v1/rpc/exec_sql`.

**Status (2026-05-11):** the project owner has decided to keep this function while the project remains pre-production. It's a deliberate dev-time convenience for running ad-hoc SQL through PostgREST during local iteration. If the project ever ships a public user-facing surface, **drop or lock down this function first** — at that point the anon key becomes a real attack vector.

**If/when locking it down later:**
```sql
-- Either drop the function entirely:
DROP FUNCTION public.exec_sql(text);

-- Or revoke public exec and lock it down:
REVOKE EXECUTE ON FUNCTION public.exec_sql(text) FROM anon, authenticated, public;
ALTER FUNCTION public.exec_sql(text) SECURITY INVOKER;
```

When migrating to a non-Supabase Postgres, **do not port this function**. It exists because Supabase's PostgREST exposes RPC over `/rest/v1/rpc/*`; outside Supabase the attack surface (PostgREST) disappears, but the function itself is still a footgun.

### MEDIUM: RLS disabled on `specialty_yearly_avg`

7 of 8 tables have RLS enabled. `specialty_yearly_avg` does not:

```sql
ALTER TABLE public.specialty_yearly_avg ENABLE ROW LEVEL SECURITY;
-- + add a policy, e.g. CREATE POLICY "public read" ON ... FOR SELECT USING (true);
```

This is the table that's not in the migration history either — likely created ad-hoc. Treat as a tech-debt item to clean up regardless of migration decision.

### LOW: `function_search_path_mutable` on `public.exec_sql`

Already covered by dropping the function.

### Performance advisor findings

- **2 unused indexes on `lstm_forecasts`**: `idx_lstm_state`, `idx_lstm_composite`. These index columns the app never queries on — candidates for `DROP INDEX`. Net storage savings ~1 MB.
- **Auth DB connection strategy** is absolute (10 connections). Not load-bearing since auth is unused.

---

## 7. Migration target options

This is a 29 MB read-mostly database. Any Postgres host fits.

### Option A — Neon (free tier, recommended)

**Cost: $0**. The Neon free tier covers 0.5 GB; this DB is 29 MB.

**Effort:** half a day.
1. Create Neon project (Postgres 17)
2. `pg_dump` from Supabase → `psql` into Neon
3. Recreate 5 extensions (Neon supports `pg_stat_statements`, `pgcrypto`, `uuid-ossp`; **drop `supabase_vault`**)
4. Drop `public.exec_sql`
5. Swap `DATABASE_URL` in whatever backend runs queries

**Pros:** scale-to-zero is ideal for this workload (the model outputs are static between training runs — the DB will idle for days). Branching for staging is a freebie.
**Cons:** cold-start ~500ms after idle.

### Option B — Self-hosted SQLite (yes, really)

Read-mostly, denormalized (no FKs), <100k rows per table, no concurrent writers. **SQLite would handle this trivially.**

**Cost: $0.** File-based.

**Effort:** 1 day to write a `pgloader`-style script (or `pg_dump --inserts` then sed-script the syntax differences) and adapt the app's query layer. The 5 extensions don't matter — none are core to the queries.

**Pros:** zero infrastructure. The whole DB is one file you can sync via Git or rsync. Perfect for an analytics serving layer.
**Cons:** SQL dialect differences (no `bigserial`, slightly different EXPLAIN). If `pgcrypto` / `uuid-ossp` are actually used by app code, you'd need to swap them for SQLite UUID generation.

### Option C — DuckDB

Same logic as SQLite but with analytical OLAP performance. Particularly good for the `lstm_forecasts` percentile lookups (62k rows × multi-column predicates).

**Cost: $0.** File-based.

**Effort:** 1 day. `duckdb -c "INSTALL postgres; LOAD postgres; CALL postgres_attach('postgres://...'); CREATE TABLE lstm_forecasts AS SELECT * FROM postgres_db.public.lstm_forecasts;"` ingests the whole DB in seconds. Or export to Parquet and ingest.

**Pros:** dramatically faster analytical queries than Postgres for this shape of data. Single-file artifact. Works in-browser via WASM if you ever want to embed it.
**Cons:** not a transactional DB. If the app does writes (it doesn't appear to), this isn't the answer.

### Option D — Railway Postgres / RDS / Cloud SQL

Standard managed Postgres. Overkill for 29 MB but simple.

**Cost:** $5–15/mo.

**Effort:** half a day. Same dump-and-restore as Option A.

### Option E — Local Postgres for dev, anything cheap for prod

Same as the CoverDriveCricket playbook. Docker `postgres:17-alpine` locally for development.

### Cost comparison summary

| Target | Monthly cost | Annual cost | Suits this workload? |
|---|---:|---:|---|
| **Supabase (current)** | $0 (free tier) | $0 | ✅ but exec_sql risk |
| Neon | $0 | $0 | ✅ best fit |
| DuckDB (file-based) | $0 | $0 | ✅✅ analytics-perfect |
| SQLite (file-based) | $0 | $0 | ✅ if app tolerates dialect |
| Railway Postgres | $5–15 | $60–180 | overkill |
| Self-hosted EC2 | $0–5 | $0–60 | overkill |

**If cost is the only driver**, you're already on free Supabase and there's no money to save. **If security is the driver, drop `public.exec_sql` immediately regardless of migration decision.** If portability / no-vendor-lockin is the driver, **DuckDB or SQLite** turn this into a file you can deploy anywhere.

---

## 8. Migration playbook (recommended: Neon free tier, or DuckDB for max portability)

### Path A — Neon

```bash
# 1. Drop the dangerous function FIRST (do this even if you don't migrate)
psql "$SUPABASE_DATABASE_URL" -c "DROP FUNCTION IF EXISTS public.exec_sql(text);"

# 2. Dump
pg_dump \
  --host=db.zdkoniqnvbklxtsviikl.supabase.co \
  --port=5432 \
  --username=postgres \
  --dbname=postgres \
  --format=custom \
  --no-owner --no-acl \
  --schema=public \
  --file=medicare-$(date +%Y%m%d).dump

# 3. Provision Neon Postgres 17. Grab the new DATABASE_URL.

# 4. Restore
pg_restore \
  --dbname="$NEON_DATABASE_URL" \
  --no-owner --no-acl \
  medicare-20260511.dump

# 5. Recreate extensions
psql "$NEON_DATABASE_URL" <<'SQL'
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
SQL

# 6. Verify
psql "$NEON_DATABASE_URL" -c "SELECT 'lstm_forecasts' AS t, COUNT(*) FROM public.lstm_forecasts
UNION ALL SELECT 'stage1', COUNT(*) FROM public.stage1_allowed_amounts
UNION ALL SELECT 'stage2', COUNT(*) FROM public.stage2_oop_estimates;"
# Expect: 62703 / 33318 / 23424

# 7. Update app DATABASE_URL, redeploy
# 8. Pause Supabase project after a 2-week burn-in
```

### Path B — DuckDB (analytics-optimized)

```bash
# 1. Export every table to Parquet from Supabase
mkdir -p medicare_export
for t in lookup_labels stage1_allowed_amounts stage2_oop_estimates lstm_forecasts \
         state_summary model_metrics feature_importances specialty_yearly_avg; do
  duckdb -c "INSTALL postgres; LOAD postgres;
             COPY (SELECT * FROM postgres_scan('$SUPABASE_DSN', 'public', '$t'))
             TO 'medicare_export/${t}.parquet';"
done

# 2. Build single DuckDB file
duckdb medicare.duckdb <<'SQL'
CREATE TABLE lookup_labels         AS SELECT * FROM 'medicare_export/lookup_labels.parquet';
CREATE TABLE stage1_allowed_amounts AS SELECT * FROM 'medicare_export/stage1_allowed_amounts.parquet';
CREATE TABLE stage2_oop_estimates  AS SELECT * FROM 'medicare_export/stage2_oop_estimates.parquet';
CREATE TABLE lstm_forecasts        AS SELECT * FROM 'medicare_export/lstm_forecasts.parquet';
CREATE TABLE state_summary         AS SELECT * FROM 'medicare_export/state_summary.parquet';
CREATE TABLE model_metrics         AS SELECT * FROM 'medicare_export/model_metrics.parquet';
CREATE TABLE feature_importances   AS SELECT * FROM 'medicare_export/feature_importances.parquet';
CREATE TABLE specialty_yearly_avg  AS SELECT * FROM 'medicare_export/specialty_yearly_avg.parquet';
SQL

# 3. Total file size will be ~5-10 MB (Parquet is denser than Postgres heap)
# 4. Ship medicare.duckdb with the app, query via duckdb-python / duckdb-wasm
```

---

## 9. Decision matrix

| Factor | Stay on Supabase | Move |
|---|---|---|
| `public.exec_sql` is a critical security hole | ⚠️ fix in place | ✅ removed by default |
| Read-mostly, denormalized, tiny data | ✅ overkill but free | ✅✅ DuckDB-perfect |
| `specialty_yearly_avg` schema drift (not in migrations) | tech debt | clean up during migration |
| Cost | $0 currently | $0 on Neon or file-based |
| Vendor lockin concern | yes | no |
| Existing tooling tied to Supabase | unknown — check app code | — |

**Recommendation:** drop `public.exec_sql` **today** (regardless of migration). After that, the workload is genuinely a "ship a Parquet/DuckDB file with the app" candidate — but if there's any operational complexity to switching, Neon's free tier is the path of least resistance.

---

*Generated 2026-05-11 from direct Supabase MCP inventory. Re-run the inventory queries to refresh.*
