// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.timeseries;

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
}