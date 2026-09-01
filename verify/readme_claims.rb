# Check the prose against the tables it is describing.
#
# The README and notes/METHODS.md were typed by hand from reports/*.csv. Nothing
# connects them: regenerate the experiments, get a slightly different number,
# and the sentence keeps the old one. That is the cheapest way for a repository
# like this one to end up lying, and it is invisible to every test here, because
# the tests never read the README.
#
# So every headline figure in the prose is looked up in the file it came from,
# and the two have to agree to within the rounding the sentence was written
# with. Two claims are not numbers in a CSV and are checked against the code
# instead: the replication count, and the number of tests.
#
# Ruby's standard library only. Run: ruby verify/readme_claims.rb <repo root>

require "csv"

ROOT = ARGV[0] || "."
$failures = 0

def report(name)
  CSV.read(File.join(ROOT, "reports", name), headers: true)
end

def text(name)
  # Explicit UTF-8. Ruby 2.6 opens files as US-ASCII by default, and the
  # README carries a non-ASCII glyph in the demo link, which makes every
  # later regex raise instead of simply not matching.
  File.read(File.join(ROOT, name), encoding: "UTF-8")
end

# Prose writes numbers with currency, thousands separators, percent signs,
# markdown bold and a typographic minus. None of that changes the value.
def num(s)
  s.to_s.tr("−", "-").delete("$,%*×").to_f
end

PEEKING = report("peeking.csv")
CUPED = report("cuped.csv")
POST = report("cuped_post_treatment_bias.csv")
TAU = report("msprt_tau.csv")
LALONDE = report("lalonde.csv")
BALANCE = report("lalonde_balance.csv")
OVERLAP = report("lalonde_overlap.csv")

README = text("README.md")
METHODS = text(File.join("notes", "METHODS.md"))

def peek(rule_prefix, col)
  row = PEEKING.find { |r| r["rule"].start_with?(rule_prefix) }
  abort "peeking.csv has no rule starting #{rule_prefix.inspect}" if row.nil?
  row[col].to_f
end

def lalonde_rows
  LALONDE.reject { |r| r["att"].to_s.strip.empty? }
end

def adjusted
  lalonde_rows.reject do |r|
    ["randomised experiment", "naive difference"].include?(r["method"])
  end
end

def truth
  lalonde_rows.find { |r| r["method"] == "randomised experiment" }
end

def naive(controls)
  lalonde_rows.find { |r| r["method"] == "naive difference" && r["controls"] == controls }
end

def closest
  adjusted.min_by { |r| r["abs error"].to_f }
end

def overlap(controls, col)
  OVERLAP.find { |r| r["controls"] == controls }[col].to_f
end

# Compare one number lifted out of the prose against the value it describes.
def claim(label, source, pattern, expected, tol)
  m = source.match(pattern)
  if m.nil?
    puts "  FAIL #{label}: the prose no longer says this at all"
    $failures += 1
    return
  end
  got = m.captures.map { |c| num(c) }
  want = Array(expected)
  unless got.length == want.length
    puts "  FAIL #{label}: matched #{got.length} numbers, expected #{want.length}"
    $failures += 1
    return
  end
  bad = got.zip(want).reject { |g, w| (g - w).abs <= tol }
  if bad.empty?
    puts format("  ok   %-46s %s", label, got.map { |g| g.to_s }.join(", "))
  else
    bad.each do |g, w|
      puts format("  FAIL %-46s prose says %s, the file says %s", label, g, w)
    end
    $failures += 1
  end
end

# A markdown table row: "| label | 1 | 2 | 3 |" as its cells.
def row_cells(source, label_pattern)
  line = source.lines.find { |l| l.start_with?("|") && l =~ label_pattern }
  return nil if line.nil?
  line.strip.sub(/\A\|/, "").sub(/\|\z/, "").split("|").map(&:strip)
end

def table_row(label, source, label_pattern, expected, tol)
  cells = row_cells(source, label_pattern)
  if cells.nil?
    puts "  FAIL #{label}: that table row is gone"
    $failures += 1
    return
  end
  got = cells[1..-1].map { |c| num(c) }
  if got.length != expected.length
    puts "  FAIL #{label}: #{got.length} columns, the file has #{expected.length}"
    $failures += 1
    return
  end
  bad = got.zip(expected).reject { |g, w| (g - w).abs <= tol }
  if bad.empty?
    puts format("  ok   %-46s %s", label, got.join(", "))
  else
    bad.each do |g, w|
      puts format("  FAIL %-46s prose says %s, the file says %s", label, g, w)
    end
    $failures += 1
  end
end

puts "README.md against reports/"
# A percentage written to one decimal place can be half a hundredth out.
PCT = 0.051
claim("peeking inflates the error rate to", README,
      /nominal 5% test into a\s+([\d.]+)% one/,
      peek("peek daily, stop", "type-I error") * 100, PCT)
claim("and the corrections pull it back to", README,
      /Pocock to ([\d.]+)%, mSPRT to ([\d.]+)%/,
      [peek("peek daily, Pocock", "type-I error") * 100,
       peek("peek daily, mSPRT", "type-I error") * 100], PCT)
claim("the power all three rules reach", README,
      /dropping from ([\d.]+) to ([\d.]+) and ([\d.]+)/,
      [peek("fixed horizon", "power"), peek("peek daily, Pocock", "power"),
       peek("peek daily, mSPRT", "power")], 0.0006)
claim("the fixed-horizon test is calibrated at", README,
      /fixed-horizon test errs ([\d.]+)% of the time/,
      peek("fixed horizon", "type-I error") * 100, PCT)
claim("power at fixed horizon against the rest", README,
      /power, ([\d.]+)% at fixed horizon against ([\d.]+)% and ([\d.]+)%/,
      [peek("fixed horizon", "power") * 100, peek("peek daily, Pocock", "power") * 100,
       peek("peek daily, mSPRT", "power") * 100], PCT)
claim("the randomised benchmark and its error", README,
      /\*\*\+\$([\d,]+)\*\* \(SE \$([\d,]+)\)/,
      [truth["att"].to_f, truth["se"].to_f], 0.5)
claim("what the naive comparison returns", README,
      /returns -\$([\d,]+) on CPS and -\$([\d,]+) on PSID/,
      [naive("cps")["att"].to_f.abs, naive("psid")["att"].to_f.abs], 0.5)
claim("how much of the PSID control pool is usable", README,
      /PSID keeps ([\d,]+) of ([\d,]+)\s*\n?\s*controls/,
      [overlap("psid", "controls inside treated PS range"),
       overlap("psid", "n control")], 0.5)
claim("the heaviest single IPW weight", README,
      /IPW weight of ([\d.]+)/, overlap("psid", "max control weight ps/(1-ps)"), 0.05)
claim("the span of the adjusted estimates", README,
      /\$([\d,]+) to \$([\d,]+), with the true/,
      [adjusted.map { |r| r["att"].to_f }.min,
       adjusted.map { |r| r["att"].to_f }.max], 0.5)
claim("the closest estimate and what it misses by", README,
      /\$([\d,]+), off by \$([\d,]+)/,
      [closest["att"].to_f, closest["abs error"].to_f], 0.5)
claim("CUPED on a mediator, and its error bar", README,
      /estimate comes back ([\d.]+)\.[\s\S]{0,60}?stays at ([\d.]+)/,
      [POST[-1]["CUPED estimate"].to_f, POST[-1]["CUPED SE (unchanged)"].to_f], 0.0006)

# The CUPED table appears in both files with the same columns, so the rho values
# are read off the header row rather than assumed.
[["README.md", README], ["notes/METHODS.md", METHODS]].each do |name, src|
  rhos = row_cells(src, /corr\(X, Y\)/)[1..-1].map(&:to_f)
  rows = rhos.map { |rho| CUPED.find { |r| r["corr(X, Y)"].to_f == rho } }
  if rows.any?(&:nil?)
    puts "  FAIL #{name}: the CUPED table has a correlation cuped.csv does not"
    $failures += 1
    next
  end
  table_row("#{name}: measured variance reduction", src, /measured reduction/,
            rows.map { |r| r["variance reduction"].to_f }, 0.0006)
  table_row("#{name}: the rho^2 it is compared against", src, /predicted \(/,
            rows.map { |r| r["predicted (rho^2)"].to_f }, 1e-9)
end

puts "\nnotes/METHODS.md against reports/"
PEEKING.each do |r|
  # Match on the whole rule name. Three of the four begin "peek daily", so
  # matching on the leading words alone selects the first of them every time
  # and silently compares one row against another row's numbers.
  head = r["rule"]
  table_row("the #{head} row", METHODS, /\A\| *(\*\*)?#{Regexp.escape(head)} *\|/,
            [r["type-I error"].to_f * 100, r["power"].to_f * 100,
             r["avg n/arm at stop"].to_f], PCT)
end
table_row("mSPRT power at each tau", METHODS, /\A\| power \|/,
          TAU.map { |r| r["power"].to_f * 100 }, PCT)
table_row("the tau grid it sweeps", METHODS, /tau \/ true effect/,
          TAU.map { |r| r["tau / true effect"].to_f }, 0.005)
table_row("plain difference under mediation", METHODS, /plain difference in means/,
          POST.map { |r| r["plain diff-in-means"].to_f }, 0.0006)
table_row("what CUPED returns instead", METHODS, /\*\*CUPED estimate\*\*/,
          POST.map { |r| r["CUPED estimate"].to_f }, 0.0006)
table_row("and its unchanging standard error", METHODS, /CUPED standard error/,
          POST.map { |r| r["CUPED SE (unchanged)"].to_f }, 0.0006)
BALANCE.each do |r|
  next unless METHODS =~ /\A\| #{Regexp.escape(r["covariate"])} \|/
  table_row("balance on #{r['covariate']}", METHODS,
            /\A\| #{Regexp.escape(r["covariate"])} \|/,
            [r["|before|"].to_f, r["|after|"].to_f], 0.0006)
end
claim("the boundary stops earlier than the horizon", METHODS,
      /([\d,]+) users instead of\s+([\d,]+)/,
      [peek("peek daily, Pocock", "avg n/arm at stop"),
       peek("fixed horizon", "avg n/arm at stop")], 0.5)
claim("one specification change moves an estimator", METHODS,
      /from \$([\d,]+) to \$([\d,]+)/,
      [LALONDE.find { |r| r["method"] == "AIPW (linear)" && r["controls"] == "psid" }["att"].to_f,
       LALONDE.find { |r| r["method"] == "AIPW (dehejia-wahba)" && r["controls"] == "psid" }["att"].to_f],
      0.5)
claim("how many adjusted estimates there are", METHODS,
      /across the (\d+) adjusted estimates/, adjusted.length, 0)
claim("and how wide they range", METHODS,
      /estimates still range (\d+)×/,
      (adjusted.map { |r| r["att"].to_f }.max / adjusted.map { |r| r["att"].to_f }.min).round,
      0)
claim("one control standing in for many", METHODS,
      /([\d.]+), meaning one person stands in for (\d+)/,
      [overlap("psid", "max control weight ps/(1-ps)"),
       overlap("psid", "max control weight ps/(1-ps)").ceil], 0.05)

puts "\nprose against the code it describes"
peeking_py = text(File.join("src", "abcausal", "experiments", "peeking.py"))
reps = peeking_py.match(/^N_REPS = ([\d_]+)/)[1].delete("_").to_i
claim("how many replications were simulated", README,
      /([\d,]+) simulated A\/A tests/, reps, 0)
n_tests = text(File.join("tests", "test_abcausal.py")).scan(/^def test_/).length
claim("how many tests assert the claims", README, /^(\d+) tests\./, n_tests, 0)

if $failures > 0
  puts "\n#{$failures} claims in the prose no longer match the files they came from"
  exit 1
end
puts "\nevery published figure in the prose matches the file it was read off"
