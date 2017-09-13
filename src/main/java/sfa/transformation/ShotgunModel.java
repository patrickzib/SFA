// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import sfa.timeseries.TimeSeries;

/**
 * The Shotgun Model as published in:
 * <p>
 * Schäfer, P.: Towards time series classification without human preprocessing.
 * In Machine Learning and Data Mining in Pattern Recognition,
 * pages 228–242. Springer, 2014.
 */
public class ShotgunModel {

  public int length;
  public boolean normMean;
  public TimeSeries[] samples;

  /**
   * Create a Shotgun model.
   *
   * @param length   break the query into disjoint subsequences of length length.
   * @param normMean set to true, if mean should be set to 0 for a window
   * @param samples  the time series to be used by the 1-NN classifier
   */
  public ShotgunModel(int length, boolean normMean, TimeSeries[] samples) {
    this.length = length;
    this.normMean = normMean;
    this.samples = samples;
  }

}
