# Roadmap

Planned extensions of the analysis, in the order they are worth doing. Each item
lists what would be added and what question it answers. Items are deliberately
scoped so that every step keeps the analysis honest about what it does and does
not correct for.

## 1. Additional signal-shape models

Add a Crystal Ball signal component (radiative tail) and compare it with the
nominal common-mean double Gaussian on both charge-separated spectra. Assign the
difference as a signal-shape systematic if the alternative describes the data
acceptably.

*Answers:* how much of the raw asymmetry depends on the signal parameterisation.

## 2. Consolidated results section

Add summary tables to the report: selection yields, fitted yields per charge,
uncertainty budget, and a single results overview table.

*Answers:* what a reader should take away without reading the whole document.

## 3. Selection systematics

Vary the baseline selection (vertex quality, flight distance, kaon PID) and, once
the classifier is useful, the classifier threshold. Track the yield and asymmetry
variations.

*Answers:* whether the asymmetry is stable against the selection, which is
currently untested.

## 4. Classifier improvements

Train on a looser preselection so the classifier has residual background to
separate, add a train/validation/test split, overtraining checks, and compare
gradient-boosted trees with the Random Forest.

*Answers:* whether a classifier can add anything on top of the baseline cuts.
The current sideband-labelled Random Forest cannot (AUC = 0.53).

## 5. Advanced statistical treatment

Move from binned to unbinned extended maximum-likelihood fits, and add
likelihood-ratio based significance estimates.

*Answers:* how much statistical information the binning currently discards.

## Longer term

- Sideband subtraction inside Dalitz regions, instead of raw candidate counting.
- Efficiency-corrected asymmetries.
- Combination of the MagnetUp and MagnetDown samples to cancel detector-induced
  charge asymmetries.
- Control-sample validation.
- Amplitude-analysis-inspired studies of the Dalitz structure.

## Scope discipline

The project stays experimentally realistic, conservative, modular and
reproducible. It avoids unnecessary deep-learning complexity, optimisation
without physics motivation, and any claim of physics sensitivity that the
corrections applied do not support.
