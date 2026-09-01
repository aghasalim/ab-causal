-- Recompute the LaLonde aggregates the README quotes, in SQLite.
--
-- The prose in the README makes four group-level claims about
-- reports/lalonde.csv: that there are 20 adjusted estimates, that they span
-- $237 to $3,843, that the closest one misses by $31, and that the naive
-- comparison is out by more than five times the effect and in the wrong
-- direction. Every one of those is an aggregate over the table, and the table
-- was written by pandas in src/abcausal/experiments/lalonde.py. Nothing
-- recomputed the aggregates.
--
-- This derives them with nothing but SQL, and rederives the per-row "abs error"
-- column by joining every row against the randomised row in the same file, so a
-- mistake in the pandas would have to be repeated here to survive. Rows whose
-- verdict is FAIL are what verify/verify.sh looks for.
--
-- Run: sqlite3 -init verify/lalonde.sql :memory: ""

.mode csv
.headers off
.import --csv reports/lalonde.csv lalonde

-- The randomised experiment is the benchmark every other row is scored against.
-- It is a row of this same file, not a constant typed in here.
CREATE TEMP VIEW truth AS
    SELECT CAST(att AS REAL) AS att
    FROM lalonde
    WHERE method = 'randomised experiment';

CREATE TEMP VIEW est AS
    SELECT controls, spec, method,
           CAST(att AS REAL)         AS att,
           CAST("abs error" AS REAL) AS published_error,
           (SELECT att FROM truth)   AS truth
    FROM lalonde
    WHERE att <> '';

CREATE TEMP VIEW adjusted AS
    SELECT * FROM est
    WHERE method NOT IN ('randomised experiment', 'naive difference');

.headers on
.mode list
.separator ' | '

-- 1. Every published abs error is the distance from the randomised benchmark.
SELECT 'abs error rederived' AS check_name,
       COUNT(*)              AS n,
       MAX(ABS(published_error - ABS(att - truth))) AS worst_gap,
       CASE WHEN MAX(ABS(published_error - ABS(att - truth))) <= 1.0
            THEN 'ok' ELSE 'FAIL' END AS verdict
FROM est;

-- 2. The adjusted estimates: how many, and the span the README quotes.
SELECT 'adjusted estimates' AS check_name,
       COUNT(*)             AS n,
       MIN(att)             AS lowest,
       MAX(att)             AS highest,
       ROUND(MAX(att) / MIN(att), 1) AS range_multiple,
       CASE WHEN COUNT(*) = 20 THEN 'ok' ELSE 'FAIL' END AS verdict
FROM adjusted;

-- 3. The closest adjusted estimate, which the README singles out by name.
SELECT 'closest adjusted' AS check_name,
       controls, spec, method, att,
       published_error AS misses_by,
       CASE WHEN published_error = (SELECT MIN(published_error) FROM adjusted)
            THEN 'ok' ELSE 'FAIL' END AS verdict
FROM adjusted
ORDER BY published_error ASC
LIMIT 1;

-- 4. The naive comparison gets the sign wrong, and by more than five times the
--    effect. Both halves of that sentence, as one aggregate per control pool.
SELECT 'naive difference' AS check_name,
       controls, att,
       ROUND(ABS(att - truth) / truth, 2) AS multiples_of_truth,
       CASE WHEN att < 0 AND truth > 0 AND ABS(att - truth) / truth > 5.0
            THEN 'ok' ELSE 'FAIL' END AS verdict
FROM est
WHERE method = 'naive difference'
ORDER BY controls;

-- 5. Every control pool crossed with every propensity specification carries the
--    same five adjusted methods. A group short of one would shrink the span
--    above without anything else noticing.
SELECT 'methods per cell' AS check_name,
       controls, spec, COUNT(*) AS n_methods,
       CASE WHEN COUNT(*) = 5 THEN 'ok' ELSE 'FAIL' END AS verdict
FROM adjusted
GROUP BY controls, spec
ORDER BY controls, spec;
