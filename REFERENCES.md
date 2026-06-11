# References

The monitor is deliberately built from established methods rather than novel machinery. Each
component maps to a primary source:

### Structural stability / fluctuation testing (the monitor form)
- Giacomini, R. & Rossi, B. (2009). *Detecting and Predicting Forecast Breakdowns.* Review of
  Economic Studies 76(2), 669–705.
- Giacomini, R. & Rossi, B. (2010). *Forecast Comparisons in Unstable Environments.* Journal of
  Applied Econometrics 25(4), 595–620.

### Sequential / path-wise change monitoring (the boundary — `csw_monitor.py`)
- Chu, C.-S. J., Stinchcombe, M. & White, H. (1996). *Monitoring Structural Change.*
  Econometrica 64(5), 1045–1065.

### Adaptive windowing (`adaptive_window.py`)
- Bifet, A. & Gavaldà, R. (2007). *Learning from Time-Changing Data with Adaptive Windowing
  (ADWIN).* Proceedings of the SIAM International Conference on Data Mining.

### Selection bias / multiple testing in finance (`dsr_pbo.py`)
- Bailey, D. H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection
  Bias, Backtest Overfitting and Non-Normality.* Journal of Portfolio Management 40(5), 94–107.
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017). *The Probability of Backtest
  Overfitting.* Journal of Computational Finance 20(4), 39–69.

### Discovery thresholds for factor research
- Harvey, C. R., Liu, Y. & Zhu, H. (2016). *…and the Cross-Section of Expected Returns.* Review of
  Financial Studies 29(1), 5–68. (Source of the 3.0 discovery t-bar.)

### Heteroskedasticity- and autocorrelation-consistent (HAC) variance
- Newey, W. K. & West, K. D. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3), 703–708.

### Factor-decay context
- McLean, R. D. & Pontiff, J. (2016). *Does Academic Research Destroy Stock Return Predictability?*
  Journal of Finance 71(1), 5–32.

### Data
- Kenneth R. French Data Library (public): factor return series used by the demos.
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
