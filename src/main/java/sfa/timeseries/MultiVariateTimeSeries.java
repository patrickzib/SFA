// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.timeseries;

import java.util.Arrays;

public class MultiVariateTimeSeries {

  public TimeSeries[] timeSeries;
  public double label = -1;

  public MultiVariateTimeSeries(TimeSeries[] timeSeries, double label) {
    this.timeSeries = timeSeries;
    this.label = label;
  }

  public int getDimensions() {
    return this.timeSeries.length;
  }

  public double getLabel() {
    return this.label;
  }

  /**
   * Get a subsequence starting at offset of windowSize.
   *
   * @param offset
   * @param windowSize
   * @return
   */
  public MultiVariateTimeSeries getSubsequence(int offset, int windowSize) {
    MultiVariateTimeSeries mts = new MultiVariateTimeSeries(new TimeSeries[this.timeSeries.length], this.label);
    for (int i = 0; i < mts.timeSeries.length; i++) {
      TimeSeries t = this.timeSeries[i];
      mts.timeSeries[i] = t.getSubsequence(offset, windowSize);
    }
    return mts;
  }

  public int getLength() {
    if (this.timeSeries != null) {
      return this.timeSeries[0].getLength();
    }
    return 0;
  }
}