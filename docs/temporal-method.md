# Quality-aware bimanual motion annotation

The new annotation layer describes **observed wrist motion**, not object contact,
grasp success, task correctness, or intent. Existing 3D poses and grip scalars are
upstream inputs; they are not newly collected ground truth. Every new interval
starts as `machine_proposed` and requires review before becoming a verified label.

## Ontology

| State | Evidence represented |
| --- | --- |
| `quiescent` | Both wrist speeds below the soft motion threshold. This can include holding an object or moving fingers. |
| `left_dominant` | Left wrist moves while right wrist is comparatively stationary. |
| `right_dominant` | Right wrist moves while left wrist is comparatively stationary. |
| `bilateral_coordinated` | Both wrists move with similar instantaneous directions and reasonably balanced speeds. This does not establish a shared goal. |
| `bilateral_independent` | Both wrists move without the directional/speed evidence required by the coordinated state. It does not assert independently intended actions. |
| `unknown` | Tracking is unusable, quality is below threshold, or wrist speed is nonfinite/negative. |

Speeds are measured from the inherited world-coordinate trajectories. Camera
motion must be handled by the feature/geometry stage; image-plane movement alone
must not be substituted without changing the documented method.

## Emission model

Each wrist has a soft movement activation
`a = sigmoid((speed - 0.055 m/s) / 0.020 m/s)`. Products of the two activations
define quiescent, unilateral, and bilateral evidence. Bilateral evidence is split
using `sigmoid((velocity_cosine - 0.65)/0.15) * sqrt(min_speed/max_speed)`.
The resulting five scores sum to one. Sensor/geometry quality `q` tempers each
score toward a uniform distribution: `p_tempered = q*p + (1-q)/5`.

These constants are transparent starting assumptions, not fitted thresholds.
Normalized scores do not imply probability calibration. `uncertainty` is the
normalized entropy of these scores; `emission_margin` is the gap between the two
largest scores. Both are review cues. Grip closure and wrist distance accompany
the proposals as evidence but do not establish object contact.

## Duration-constrained decoding

An expanded-state Viterbi decoder gives each semantic state a sequence of age
states. A label must spend at least 0.30 seconds in its age chain before changing
state; it can remain mature for arbitrarily long. A change costs 2.2 log-score
units. Dynamic programming finds the highest-scoring complete path under these
constraints. Complexity is `O(T*K*(D+K))` with `O(T*K*D)` backtrace memory, where
`K=5` and `D` is the minimum dwell in frames.

The final segment of an observed span can be shorter because its right boundary
is censored. Invalid frames, dropped frame indices, or timestamp gaps over 2.5
times the median frame interval split decoding. Unknown runs are retained as
annotation intervals; nothing is interpolated across a gap. Frame intervals and
time intervals are half-open, and the last observed frame contributes one nominal
frame period. Boundary censoring and short fragments are exported explicitly.
`end_frame` is a compatibility alias of `end_frame_exclusive`; both exclude that
frame. A segment's `qa_flags` is the union of its input-frame QA flags.

## Review policy

Frame review risk combines 50% emission entropy, 30% quality deficit, and 20%
disagreement between the independent emission decision and decoded state.
Unknown frames receive risk one. Segment risk is the mean over its frames.
Coverage is an explicit first-stage constraint: every available motion state
(including `unknown`) and every episode receives a representative when the
budget supports it. The episode–state bipartite graph has an edge for each
available combination. Deterministic maximum matching constructs a minimum-size
edge cover, using the highest-risk segment for each chosen edge. Thus a feasible
coverage budget cannot be wasted on redundant high-risk unknowns. The seeds are
ordered greedily by newly covered states/episodes, preferring new states on ties.
When the budget is too small for complete coverage, the same coverage ordering
runs over all candidates; a budget of at least the number of states still covers
every state before adding redundant states solely for episode coverage.

After coverage, remaining slots use the existing deterministic score: risk
(50%), unseen-episode coverage (22%), unseen-state coverage (16%), and separation
from already selected segments (12%, saturating at 15 seconds). Output records
identify the coverage or fill phase and the minimum complete-coverage budget.
This is a transparent sampling policy, not a trained active-learning model.
Selected records remain machine proposals, and coverage does not measure
semantic accuracy.

## Evaluation boundaries

The raw-versus-decoded transition counts quantify flicker suppression. Fewer
transitions are not evidence of better semantic accuracy; excessive smoothing
can erase short real events. Unit tests compare small Viterbi problems against
an exhaustive optimum, verify abstention/gap handling, check interval accounting,
and demonstrate review diversity. Semantic performance requires independently
adjudicated labels, boundary tolerance metrics, and an episode-held-out protocol.
Do not report the heuristic uncertainty as calibrated confidence or the inherited
pose projections as an independent pose-accuracy benchmark.
