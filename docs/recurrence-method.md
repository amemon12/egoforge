# Experimental recurrent-motion layer

Repeated tool manipulation often produces an oscillating hand trajectory. This
layer proposes **rhythmic wrist-motion bouts** from the inherited, cleaned 3D
world trajectories. It does not identify the tool, infer contact, count cuts, or
establish task completion. Its motion cycles are a separate experimental layer
from the bimanual motion-state intervals.

## Spatial and temporal evidence

The detector analyzes each wrist in overlapping five-second windows with a
one-second stride. It removes the window's mean translation and linear drift,
then computes a three-dimensional principal component. A stable sign convention
on the component makes extrema deterministic. The projected signal must have
at least 15 mm between its 5th and 95th percentiles, and the dominant component
must account for at least 55% of detrended positional variance.

The normalized overlapping autocorrelation searches periods from 0.25 to 2.0
seconds. The earliest local peak within 90% of the strongest eligible peak is
selected, subject to autocorrelation at least 0.60. This favors the fundamental
over a multiple without assuming that harmonic ambiguity is eliminated.
Subframe parabolic interpolation refines the period estimate; exported extrema
remain actual observed frame timestamps. A window must support at least 2.5
estimated cycles and at least three observed positive peaks. Peak-interval
coefficient of variation must be at most 0.30, and the observed median interval
must agree with the autocorrelation period within 25%.

Every window must consist entirely of usable frames with quality at least 0.50.
Nonfinite wrist positions, missing frame indices, or timing steps deviating by
more than 25% from the episode median reject a window. No gap interpolation or
clock correction occurs. Using world coordinates removes camera-induced image
motion from the representation; inherited tracking drift and whole-body motion
can still produce misleading recurrence and require review.

## Bouts and cycle events

At least two consecutively accepted, overlapping windows with compatible periods
form a candidate bout. Rejected windows interrupt consolidation. Repeated
observations of the same extremum are clustered, retaining an actual source
frame. Only adjacent peak pairs agreeing with the bout period within 25% become
cycle proposals. Intervals are half-open. Boundary fragments are not counted as
completed cycles. Same-hand hypotheses overlapping by more than 25% of the
shorter bout compete: the greater `coherence * sqrt(supporting_windows)` wins,
and the alternative is preserved separately with its suppression reason. This
prevents conflicting period hypotheses from being counted twice. Both wrists
may describe the same physical activity, so the
sum across sides must not be described as a count of independent actions.

`side` identifies the wrist whose recurrence was detected. `dominant_side`
compares the two wrists' detrended principal-axis extents over the bout, using a
1.25 ratio; otherwise it is `balanced`. This is amplitude dominance, not a
semantic distinction such as stabilizing hand versus tool hand.

`coherence` combines autocorrelation, explained axis variance, peak-interval
consistency, and input quality. `uncertainty = 1 - coherence` is a heuristic
review cue, not a calibrated probability. Every window, bout, and cycle remains
`machine_proposed`; accepted windows mean the numerical criteria passed, not
that a person verified the motion or its meaning. Rejected windows retain
explicit reasons for auditability.

## Validation and limits

Tests recover known sinusoidal periods, reject stationary/linear trajectories,
verify invariance to rigid rotation of world coordinates, enforce gaps and
quality gates, and check deterministic output. These are functional tests of a
detector, not semantic performance estimates on natural video. Independent
human review is required to measure precision, missed bouts, boundary error,
harmonic confusion, and sensitivity to small tool movements. The minimum extent,
duration and quality gates intentionally trade coverage for reviewable evidence.
