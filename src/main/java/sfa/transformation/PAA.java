// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import java.io.Serializable;
import sfa.timeseries.TimeSeries;

/**
 *
 * Piecewise Aggregate Approximation
 *
 * E. Keogh, K. Chakrabarti, M. Pazzani & S. Mehrotra  Dimensionality Reduction for Fast
 * Similarity Search in Large Time Series Databases.  Knowledge and Information Systems,
 * vol. 3, no. 3. pp263-286,  2000
 *
 */
public class PAA extends Representation {
  private static final long serialVersionUID = 6550901047488847653L;

  public PAA() {
    super();
  }

  /**
   * Divide the time series into l equal size segments and computes the mean
   */
  @Override
  public TimeSeries transform(TimeSeries timeSeries, int l) {
    double[] means = new double[l];
    double[] data = timeSeries.getData();
    int n = timeSeries.getLength();

    // calculate n mean values
    double sizeOfFrame = (double)n/(double)l;
    for (int i = 0; i < l; i++) {
      double mean = 0;
      double size = 0.0;

      // calculate means
      int s1 = (int)Math.floor(sizeOfFrame*i);
      int e1 = (int)Math.min(timeSeries.getLength(),Math.ceil(sizeOfFrame*(i+1)));
      for (int j = s1; j < e1; j++) {
        mean += data[j];
        size++;
      }
      means[i] = mean/(size>0?size:1);
    }

    return new TimeSeries(means);
  }

  public TimeSeries inverseTransform (TimeSeries timeSeries, int n) {

    int l = timeSeries.getLength();
    double[] means = timeSeries.getData();
    double[] data = new double[n];
    double sizeOfFrame = (double)n/(double)l;

    //  for every interval, insert the mean
    for (int i = 0; i < l; i++) {
      for (int j = (int)Math.ceil(sizeOfFrame*i); j < Math.ceil(sizeOfFrame*(i+1)); j++) {
        data[j] = means[i];
      }
    }

    return new TimeSeries(data);
  }

  public double getDistance(TimeSeries t1, TimeSeries t2, TimeSeries originalQuery, int n, double minValue) {

    int l = t1.getLength();

    double distance = 0;
    for (int i = 0; i < l; i++) {
      double value = t1.getData()[i]-t2.getData()[i];
      distance += value*value;

      // early stopping
      if (distance > minValue) {
        return Double.POSITIVE_INFINITY;
      }
    }

    return distance * (n / l);
  }
}
