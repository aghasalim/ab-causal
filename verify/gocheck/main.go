// Structural validation of everything under reports/, plus a recomputation of
// every column in those files that is derived from another column.
//
// The CSVs in reports/ are the evidence for every number in the README and the
// only thing the deployed app reads. Nothing checked that they are well formed:
// a truncated write, a column that drifted, or a NaN that leaked out of a
// division would all be invisible until someone read the table. This walks
// every file, and then rederives the columns that are functions of other
// columns, which is where a copy-paste error would land.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// TRUE_EFFECT in src/abcausal/experiments/peeking.py, and the horizon and daily
// intake that fix the largest sample an experiment can reach.
const (
	trueEffect = 0.10
	horizon    = 14
	nPerDay    = 100
)

type table struct {
	header []string
	rows   [][]string
}

func (t table) col(name string) int {
	for i, h := range t.header {
		if h == name {
			return i
		}
	}
	return -1
}

// num parses cell (row, column name). ok is false for a blank cell, which is
// how the LaLonde table marks a statistic that does not apply to that method.
func (t table) num(row int, name string) (float64, bool) {
	c := t.col(name)
	if c < 0 || c >= len(t.rows[row]) {
		return 0, false
	}
	s := strings.TrimSpace(t.rows[row][c])
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	return v, err == nil
}

func (t table) str(row int, name string) string {
	c := t.col(name)
	if c < 0 || c >= len(t.rows[row]) {
		return ""
	}
	return strings.TrimSpace(t.rows[row][c])
}

func read(path string) (table, error) {
	f, err := os.Open(path)
	if err != nil {
		return table{}, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return table{}, err
	}
	if len(rows) < 2 {
		return table{}, fmt.Errorf("only %d rows", len(rows))
	}
	return table{header: rows[0], rows: rows[1:]}, nil
}

// validate reports every structural problem in one file rather than the first,
// so a broken run is diagnosed in one pass.
func validate(path string, t table) []string {
	var problems []string

	seen := map[string]bool{}
	for _, h := range t.header {
		if strings.TrimSpace(h) == "" {
			problems = append(problems, "a column has an empty name")
		}
		if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}

	for i, row := range t.rows {
		for j, cell := range row {
			low := strings.ToLower(strings.TrimSpace(cell))
			if low == "nan" || low == "inf" || low == "-inf" || low == "infinity" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is %s", i+2, t.header[j], cell))
			}
		}
	}
	return problems
}

// near reports whether two numbers agree, with a tolerance loose enough for
// the rounding the published tables were written with.
func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

type checker struct {
	failures int
}

func (c *checker) fail(format string, args ...any) {
	fmt.Printf("    FAIL %s\n", fmt.Sprintf(format, args...))
	c.failures++
}

func (c *checker) okf(format string, args ...any) {
	fmt.Printf("    ok   %s\n", fmt.Sprintf(format, args...))
}

// cuped: the predicted column is rho^2 by definition, and the sample-saving
// column is the variance reduction as a percentage.
func (c *checker) cuped(t table) {
	before := c.failures
	for i := range t.rows {
		rho, ok1 := t.num(i, "corr(X, Y)")
		vr, ok2 := t.num(i, "variance reduction")
		pred, ok3 := t.num(i, "predicted (rho^2)")
		if !ok1 || !ok2 || !ok3 {
			c.fail("cuped.csv row %d is missing a value", i+2)
			continue
		}
		if rho < 0 || rho > 1 {
			c.fail("cuped.csv row %d: corr %g is not a correlation", i+2, rho)
		}
		if !near(pred, rho*rho, 5e-5) {
			c.fail("cuped.csv row %d: predicted %.4f is not rho^2 %.4f",
				i+2, pred, rho*rho)
		}
		if vr > 1.0 {
			c.fail("cuped.csv row %d: variance reduction %.4f exceeds 1", i+2, vr)
		}
		want := fmt.Sprintf("%.0f%%", vr*100)
		if got := t.str(i, "equiv. sample saving"); got != want {
			c.fail("cuped.csv row %d: sample saving %q, from a reduction of "+
				"%.4f that is %q", i+2, got, vr, want)
		}
	}
	if c.failures == before {
		c.okf("cuped.csv: predicted column is rho^2 and the saving column is "+
			"the reduction, all %d rows", len(t.rows))
	}
}

// post-treatment bias: the error column is the estimate minus the total effect,
// and the total effect is held at TRUE_EFFECT in every row by construction.
func (c *checker) postTreatment(t table) {
	before := c.failures
	for i := range t.rows {
		total, ok1 := t.num(i, "true TOTAL effect")
		est, ok2 := t.num(i, "CUPED estimate")
		err, ok3 := t.num(i, "CUPED error vs total")
		if !ok1 || !ok2 || !ok3 {
			c.fail("cuped_post_treatment_bias.csv row %d is missing a value", i+2)
			continue
		}
		if !near(total, trueEffect, 1e-9) {
			c.fail("cuped_post_treatment_bias.csv row %d: total effect %.4f, "+
				"the experiment holds it at %.2f", i+2, total, trueEffect)
		}
		if !near(err, est-total, 1e-4) {
			c.fail("cuped_post_treatment_bias.csv row %d: error %.4f, "+
				"estimate minus total is %.4f", i+2, err, est-total)
		}
	}
	if c.failures == before {
		c.okf("cuped_post_treatment_bias.csv: error column is estimate minus "+
			"total in all %d rows", len(t.rows))
	}
}

// peeking and the tau sweep: rates are probabilities, and no rule can stop
// later than the horizon or earlier than the first look.
func (c *checker) rates(name string, t table) {
	before := c.failures
	minN, maxN := float64(nPerDay), float64(nPerDay*horizon)
	for i := range t.rows {
		for _, col := range []string{"type-I error", "power"} {
			v, ok := t.num(i, col)
			if !ok {
				c.fail("%s row %d has no %s", name, i+2, col)
				continue
			}
			if v < 0 || v > 1 {
				c.fail("%s row %d: %s is %.4f, not a probability", name, i+2, col, v)
			}
		}
		n, ok := t.num(i, "avg n/arm at stop")
		if !ok {
			c.fail("%s row %d has no stopping sample size", name, i+2)
		} else if n < minN || n > maxN {
			c.fail("%s row %d: stops at n=%.0f, outside the %d..%d the design "+
				"allows", name, i+2, n, int(minN), int(maxN))
		}
	}
	if c.failures == before {
		c.okf("%s: all rates are probabilities and every rule stops inside "+
			"%d..%d per arm", name, int(minN), int(maxN))
	}
}

func (c *checker) tauSweep(t table) {
	before := c.failures
	for i := range t.rows {
		tau, ok1 := t.num(i, "tau")
		ratio, ok2 := t.num(i, "tau / true effect")
		if !ok1 || !ok2 {
			c.fail("msprt_tau.csv row %d is missing a value", i+2)
			continue
		}
		if !near(ratio, tau/trueEffect, 5e-3) {
			c.fail("msprt_tau.csv row %d: ratio %.2f, tau/%.2f is %.2f",
				i+2, ratio, trueEffect, tau/trueEffect)
		}
	}
	if c.failures == before {
		c.okf("msprt_tau.csv: ratio column is tau over the simulated effect, "+
			"all %d rows", len(t.rows))
	}
}

// The fixed-horizon row is the calibration the whole peeking claim rests on: it
// tests once, so it must always run to the horizon, and its error rate must be
// the nominal alpha.
func (c *checker) fixedHorizon(t table) {
	for i := range t.rows {
		if !strings.HasPrefix(t.str(i, "rule"), "fixed horizon") {
			continue
		}
		n, _ := t.num(i, "avg n/arm at stop")
		if int(n) != nPerDay*horizon {
			c.fail("peeking.csv: the fixed-horizon rule stops at n=%.0f, but "+
				"testing once means always reaching %d", n, nPerDay*horizon)
			return
		}
		e, _ := t.num(i, "type-I error")
		if e < 0.04 || e > 0.06 {
			c.fail("peeking.csv: the fixed-horizon test errs %.4f of the time, "+
				"so the harness is not calibrated and nothing measured against "+
				"it means anything", e)
			return
		}
		c.okf("peeking.csv: the fixed-horizon row reaches n=%d and errs %.4f, "+
			"so the harness is calibrated", int(n), e)
		return
	}
	c.fail("peeking.csv has no fixed-horizon row to calibrate against")
}

// LaLonde: the error column is the distance from the randomised benchmark, and
// the benchmark is a row of the same table.
func (c *checker) lalonde(t table) {
	before := c.failures
	truth, found := 0.0, false
	for i := range t.rows {
		if t.str(i, "controls") == "randomised" {
			truth, found = t.num(i, "att")
			break
		}
	}
	if !found {
		c.fail("lalonde.csv has no randomised row, so nothing can be scored")
		return
	}
	adjusted := 0
	for i := range t.rows {
		att, ok1 := t.num(i, "att")
		abs, ok2 := t.num(i, "abs error")
		if !ok1 || !ok2 {
			c.fail("lalonde.csv row %d is missing att or abs error", i+2)
			continue
		}
		if !near(abs, math.Abs(att-truth), 1.0) {
			c.fail("lalonde.csv row %d: abs error %.0f, |att - %.0f| is %.0f",
				i+2, abs, truth, math.Abs(att-truth))
		}
		m := t.str(i, "method")
		if m != "randomised experiment" && m != "naive difference" {
			adjusted++
		}
	}
	if adjusted != 20 {
		c.fail("lalonde.csv holds %d adjusted estimates, the README claims 20",
			adjusted)
	}
	if c.failures == before {
		c.okf("lalonde.csv: abs error is the distance from the randomised "+
			"$%.0f in all %d rows, and there are %d adjusted estimates",
			truth, len(t.rows), adjusted)
	}
}

// Overlap: a count of controls inside a range cannot exceed the pool it was
// counted from.
func (c *checker) overlap(t table) {
	before := c.failures
	for i := range t.rows {
		n, ok := t.num(i, "n control")
		if !ok {
			c.fail("lalonde_overlap.csv row %d has no control count", i+2)
			continue
		}
		for _, col := range []string{
			"controls inside treated PS range", "controls with PS < 0.01",
		} {
			v, ok := t.num(i, col)
			if !ok {
				c.fail("lalonde_overlap.csv row %d has no %q", i+2, col)
				continue
			}
			if v < 0 || v > n {
				c.fail("lalonde_overlap.csv row %d: %q is %.0f out of a pool "+
					"of %.0f", i+2, col, v, n)
			}
		}
		w, ok := t.num(i, "max control weight ps/(1-ps)")
		if ok && w <= 0 {
			c.fail("lalonde_overlap.csv row %d: max weight %.1f is not positive",
				i+2, w)
		}
	}
	if c.failures == before {
		c.okf("lalonde_overlap.csv: every subset count fits inside its control "+
			"pool, all %d rows", len(t.rows))
	}
}

// Balance: standardised mean differences are absolute values, so negative is a
// sign error rather than a small number.
func (c *checker) balance(t table) {
	before := c.failures
	for i := range t.rows {
		for _, col := range []string{"|before|", "|after|"} {
			v, ok := t.num(i, col)
			if !ok {
				c.fail("lalonde_balance.csv row %d has no %q", i+2, col)
				continue
			}
			if v < 0 {
				c.fail("lalonde_balance.csv row %d: %q is %.3f, an absolute "+
					"value cannot be negative", i+2, col, v)
			}
		}
	}
	if c.failures == before {
		c.okf("lalonde_balance.csv: all %d standardised differences are "+
			"non-negative", len(t.rows))
	}
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	reports := filepath.Join(*root, "reports")
	paths, err := filepath.Glob(filepath.Join(reports, "*.csv"))
	if err != nil || len(paths) == 0 {
		fmt.Fprintf(os.Stderr, "no CSVs under %s\n", reports)
		os.Exit(2)
	}
	sort.Strings(paths)

	c := &checker{}
	tables := map[string]table{}

	fmt.Printf("structure of %d files under reports/\n", len(paths))
	for _, path := range paths {
		base := filepath.Base(path)
		t, err := read(path)
		if err != nil {
			c.fail("%s unreadable: %v", base, err)
			continue
		}
		tables[base] = t
		for _, p := range validate(path, t) {
			c.fail("%s: %s", base, p)
		}
	}
	if c.failures == 0 {
		fmt.Printf("    ok   no ragged rows, duplicate columns, NaN or Inf anywhere\n")
	}

	fmt.Printf("\nderived columns, rederived from the columns they come from\n")
	want := []struct {
		file string
		fn   func(table)
	}{
		{"cuped.csv", c.cuped},
		{"cuped_post_treatment_bias.csv", c.postTreatment},
		{"peeking.csv", func(t table) { c.rates("peeking.csv", t); c.fixedHorizon(t) }},
		{"msprt_tau.csv", func(t table) { c.rates("msprt_tau.csv", t); c.tauSweep(t) }},
		{"lalonde.csv", c.lalonde},
		{"lalonde_overlap.csv", c.overlap},
		{"lalonde_balance.csv", c.balance},
	}
	for _, w := range want {
		t, ok := tables[w.file]
		if !ok {
			c.fail("reports/%s is missing, and the app reads it", w.file)
			continue
		}
		w.fn(t)
	}

	if c.failures > 0 {
		fmt.Printf("\n%d problems\n", c.failures)
		os.Exit(1)
	}
	fmt.Printf("\nreports/ is well formed and every derived column rederives\n")
}
