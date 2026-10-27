# Roadmap

Planned extensions of the analysis, in the order they are worth doing. Each item
lists what would be added and what question it answers. Items are deliberately
scoped so that every step keeps the analysis honest about what it does and does
not correct for.

## Done

- **Simultaneous $B^+$/$B^-$ fits with shared shapes**, on both magnet
  polarities, averaged with equal weights. This removed the fit-range dependence
  that separate fits had, and brought the systematic below the statistical
  uncertainty.
- **Joint model × range systematic**, instead of a quadrature sum of
  one-at-a-time shifts that are not independent; Barlow correction of the range
  shifts, validated with toys.
- **The MagnetDown $B^+$ low-mass shoulder**, tested with a toy-calibrated
  shape test on both polarities: a ~2σ fluctuation that MagnetUp does not
  reproduce.
- **Charge-ordered Dalitz variables**, toy-based goodness-of-fit p-values, a
  global p-value for the local Dalitz scan.

## 1. Per-charge momentum-scale calibration

Fit the $B$ peak position separately for each charge and polarity, and derive a
momentum-scale correction from its offset to the known $B^\pm$ mass. In separate
fits the $B^+ - B^-$ peak difference is −2.5 ± 1.2 MeV/$c^2$ in MagnetDown and
+1.8 ± 1.3 MeV/$c^2$ in MagnetUp, a sign flip with the polarity that an
uncalibrated charge-dependent scale would produce.

*Answers:* whether the mass scale differs between charges, and whether the shared
peak position of the simultaneous fit is a valid assumption. Possible with the
data in this record.

## 2. Selection systematics

Vary the baseline selection (vertex quality, flight distance, kaon PID) and track
the polarity-averaged asymmetry, with the Barlow correction since each variation
changes the sample.

*Answers:* whether the asymmetry is stable against the selection, which is
currently untested.

## 3. Signal and partially reconstructed shapes

Add a Crystal Ball signal component (radiative tail), the variation already
declared as pending in `studies/systematics/configurations.py`, and an ARGUS
shape for the partially reconstructed structure below 5200 MeV/$c^2$ with its
parameters fixed from simulation. The ntuple starts at 5050 MeV/$c^2$, which
leaves too little of that structure to constrain its shape from the data: a free
ARGUS curvature runs to its limit. Check first whether `PhaseSpaceSimulation.root`
carries reconstructed kinematics; if it does not, this needs simulation outside
the record.

*Answers:* how much of the result depends on the signal tail, and whether the fit
range can be extended to use the low-mass sideband.

## 4. Production and kaon-detection asymmetries

The polarity average cancels left–right detection asymmetries, but not the
$B^\pm$ production asymmetry or the $K^+/K^-$ detection asymmetry. Both need a
control channel with a known CP asymmetry, such as $B^\pm \to J/\psi K^\pm$,
which is not in this record.

*Answers:* what separates the raw asymmetry from a CP asymmetry. Needs data
outside the record.

## 5. Classifier improvements

Train on a looser preselection so the classifier has residual background to
separate, add a train/validation/test split, overtraining checks, and compare
gradient-boosted trees with the Random Forest.

*Answers:* whether a classifier can add anything on top of the baseline cuts.
The current sideband-labelled Random Forest cannot (AUC = 0.53).

## 6. Advanced statistical treatment

Move from binned to unbinned extended maximum-likelihood fits, and add
likelihood-ratio based significance estimates.

*Answers:* how much statistical information the binning currently discards.

## Longer term

- Background subtraction inside Dalitz regions (sPlot or per-bin fits), instead
  of raw candidate counting.
- Efficiency-corrected asymmetries across the Dalitz plane.
- Amplitude-analysis-inspired studies of the Dalitz structure.
- Extend the exploratory notebooks (`01`, `03`, `04`, `05`) to both polarities;
  they currently use MagnetDown only.

## Scope discipline

The project stays experimentally realistic, conservative, modular and
reproducible. It avoids unnecessary deep-learning complexity, optimisation
without physics motivation, and any claim of physics sensitivity that the
corrections applied do not support.
