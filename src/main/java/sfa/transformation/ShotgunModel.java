// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

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

  /**
   * Create a Shotgun model.
   *
   * @param length   break the query into disjoint subsequences of length length.
   * @param normMean set to true, if mean should be set to 0 for a window
   */
  public ShotgunModel(int length, boolean normMean) {
    this.length = length;
    this.normMean = normMean;
  }

}
