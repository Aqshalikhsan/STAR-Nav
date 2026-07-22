"""Results-export layer.

Runs the real STAR-Nav pipeline (and baselines) and writes CSVs in the exact
parameter/metric schema of the paper's result set, so a reviewer can reproduce
the *format* of every table/figure by running the runnable backends
(Mock / Gazebo). No number is synthesised: every value here is measured from a
real rollout or a real perception pass. See `grid.py` for the parameter grid
and `schema.py` for the per-category column contracts.
"""
