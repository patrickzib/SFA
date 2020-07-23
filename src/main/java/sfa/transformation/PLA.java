// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import sfa.timeseries.TimeSeries;

import java.io.Serializable;


/**
 * Implementation of Piecewise Linear Approximation-Signatur
 *
 *  Y. Morinaka, M. Yoshikawa, T. Amagasa & S. Uemura  The L-index: An Indexing Structure
 *  for Efficient Subsequence Matching in Time Sequence Databases. In Proc. of Pacific-Asian
 *  Conf. on Knowledge Discovery and Data Mining.  pp 51-60, 2001.
 *
 *  Qiuxia Chen, Lei Chen, Xiang Lian, Yunhao Liu, and Jeffrey. X. Yu, �Indexable PLA for
 *  Efficient Similarity Search�, in Proceedings of 33rd International Conference on Very
 *  Large Data Bases (VLDB'07), 2007.
 */
public class PLA extends Representation {
  private static final long serialVersionUID = -6323381580079336581L;

  public PLA() {
    super();
  }

  /**
   * Divide time series into l-even sized line segments
   *
   * Line: st = a*t +b
   * Time Series : <a1, b1, a2, b2, ...>
   */
  @Override
  public TimeSeries transform(TimeSeries timeSeries, int l) {
    int n = timeSeries.getLength();

    double[] values = new double[l];
    double[] data = timeSeries.getData();

    double sizeOfFrame = (double)n/(double)(l/2);

    // l/2-line segements using least squares
    for (int i = 0; i < l/2.0; i++) {

      double a = 0.0;
      double b = 0.0;

      double constA = (sizeOfFrame + 1) / 2.0;
      double constB = (2*sizeOfFrame + 1) / 3.0;

      int t = 1;

      // for each segment
      for (int j =  (int)Math.ceil(sizeOfFrame*i); j < Math.min(data.length,Math.ceil(sizeOfFrame*(i+1))); j++) {
        a += (t - constA) * data[j];
        b += (t - constB) * data[j];
        t++;
      }

      // TODO if sizeOfFrame == 1 => NaN
      a /= sizeOfFrame*(sizeOfFrame+1)*(sizeOfFrame-1);
      b /= sizeOfFrame*(1-sizeOfFrame);

      values[i*2] = 12*a;
      values[i*2+1] = 6*b;
    }

    return new TimeSeries(values);
  }

  public TimeSeries inverseTransform(TimeSeries timeSeries, int n) {

    int l = timeSeries.getLength();

    double[] values = new double[n];
    double[] data = timeSeries.getData();

    double sizeOfFrame = (double)n/(double)(l/2);

    // compute value for l/2-line segments
    for (int i = 0; i < l/2; i++) {
      int t = 1;
      for (int j = (int)Math.ceil(sizeOfFrame*i); j < (int)Math.ceil(sizeOfFrame*(i+1)); j++) {
        values[j] = data[(i*2)]*t + data[i*2+1];
        t++;
      }
    }

    return new TimeSeries(values);
  }

  public double getDistance(TimeSeries t1, TimeSeries t2, TimeSeries originalQuery, int n, double minValue) {

    int l = t1.getLength();

    double distance = 0;
    double segments = l/2.0;
    double frameSize = (int)((double)n/(segments));

    double constB = frameSize*(frameSize + 1);
    double constA = constB*(2*frameSize + 1) / 6.0;


    // a segment has two components
    for (int i = 0; i < l; i+=2) {

      // Line defined by s_i(t) = a_i*t +b_i
      double t1ai = t1.getData()[i];
      double t1bi = t1.getData()[i+1];
      double t2ai = t2.getData()[i];
      double t2bi = t2.getData()[i+1];

      double t3ai = (t1ai-t2ai);
      double t3bi = (t1bi-t2bi);

      distance += constA*t3ai*t3ai + constB*t3ai*t3bi + frameSize*t3bi*t3bi;

      // early stopping
      if (distance > minValue) {
        return Double.POSITIVE_INFINITY;
      }
    }
    return distance;
  }
}
