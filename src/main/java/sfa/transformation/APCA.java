// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import sfa.index.SortedListMap;
import sfa.timeseries.TimeSeries;

/**
 *
 * Implementation of Adaptive Piecewise Constant Approximation
 *
 *  E. Keogh, K. Chakrabarti, M. Pazzani & S. Mehrotra  Locally Adaptive Dimensionality
 *  Reduction for Indexing Large Time Series Databases. In Proc. of the ACM SIGMOD Int'l
 *  Conf. on Management of Data. pp 188-288. 2001.
 */
public class APCA extends Representation implements Serializable {
  private static final long serialVersionUID = -5161865257245093223L;

  private transient DWT dtw = new DWT();

  /**
   * Computes the mean over n/2 segements of a time series of variable length
   */
  @SuppressWarnings("unchecked")
  public TimeSeries transform(TimeSeries timeSeries, int n) {

    int t1Size = timeSeries.getLength();
    int m = n / 2;

    if (n > t1Size) {
      throw new IllegalArgumentException("Too many coefficients selected");
    }

    // transform into wavelet representation
    TimeSeries waveletRepresentation = dtw.transform(timeSeries, t1Size);
    SortedListMap<Double, Integer> firstNKoefficients = new SortedListMap<Double, Integer>(n); // TODO repace by priorityquery???

    // sort coefficients in descending order
    for (int i = 1; i < waveletRepresentation.getLength(); i++) {
      int divisor = i>1? (int)(Math.log(i)/Math.log(2.0)) : 0;
      double norm = Math.pow(2, divisor/2.0);
      firstNKoefficients.put(-Math.abs(waveletRepresentation.getData()[i] / norm), i);
    }

    // set all other coefficients to 0 within the signal
    double[] representation = new double[t1Size];
    while (!firstNKoefficients.isEmpty()) {
      double key = firstNKoefficients.firstKey();
      int position = firstNKoefficients.removeFirst(key);
      representation[position] = waveletRepresentation.getData()[position];
    }
    waveletRepresentation.setData(representation);
    TimeSeries truncatedSignal = dtw.inverseTransform(waveletRepresentation, timeSeries.getLength());

    // join segments with identical value
    List<Integer> keys = new ArrayList<Integer>(3*n);
    List<Double> values = new ArrayList<Double>(3*n);
    double oldValue = truncatedSignal.getData()[1];
    for (int i = 1; i < truncatedSignal.getLength(); i++) {
      // new segment?
      double currentValue = truncatedSignal.getData()[i];
      if (oldValue!=currentValue) {
        keys.add(i);
        values.add(oldValue);
        oldValue = currentValue;
      }
    }

    // set end point
    keys.add(truncatedSignal.getLength());
    values.add(truncatedSignal.getData()[truncatedSignal.getLength()-1]);

    // compute mean over intervals
    int start = 0;
    int pos2 = 0;
    for (int end : keys) {
      // compute mean
      double realMean = 0.0;
      for (int i = start; i < end; i++) {
        realMean += timeSeries.getData()[i];
      }
      realMean /= (double)(end - start);
      values.set(pos2, realMean);
      start = end;
      pos2++;
    }

    //  Join intervals that lead to smallest increase in reconstruction error
    while (keys.size() > m) {
      double minReconstructionError = Double.POSITIVE_INFINITY;
      int minJoinInterval = 0;
      double reconstructionErrorLeft = 0;
      double reconstructionErrorRight = calcReconstructionError(timeSeries, keys, values, 0);
      double reconstructionErrorBoth = 0;

      // compute reconstruction error for each interval
      for (int pos = 0; pos < keys.size()-1; pos++) {
        reconstructionErrorLeft = reconstructionErrorRight;
        reconstructionErrorRight = calcReconstructionError(timeSeries, keys, values, pos+1);
        reconstructionErrorBoth = calcJointReconstructionError(timeSeries, keys, values, pos);
        double changeOfReconstructionError = reconstructionErrorBoth-(reconstructionErrorLeft + reconstructionErrorRight);
        if (changeOfReconstructionError < minReconstructionError) {
          minReconstructionError = changeOfReconstructionError;
          minJoinInterval = pos;
        }
      }

      // perform merge
      int startFirst = minJoinInterval > 0? keys.get(minJoinInterval-1) : 0;
      int endFirst = keys.get(minJoinInterval);
      int endSecond = keys.get(minJoinInterval+1);
      double mean = (((endFirst-startFirst) * values.get(minJoinInterval) +  (endSecond-endFirst) * values.get(minJoinInterval+1))) / (double)(endSecond-startFirst);
      values.set(minJoinInterval+1, mean);
      keys.remove(minJoinInterval);
      values.remove(minJoinInterval);
    }

    // obtain APCA representation as Tupel (value, interval-end)
    double [] apca = new double[n];
    int i = 1;
    for (double key : keys) {
      apca[i] = key;
      i+=2;
    }
    i = 0;
    for (double value : values) {
      apca[i] = value;
      i+=2;
    }

    if (keys.size()<m) {
      for (int j = 2*keys.size(); j < n; j+=2) {
        apca[j+1]=truncatedSignal.getLength();
        apca[j]=truncatedSignal.getData()[truncatedSignal.getLength()-1];
      }
    }

    if (apca[apca.length-1]<t1Size-1) {
      System.err.println("Error!");
    }

    //System.out.println(Arrays.toString(apca));
    return new TimeSeries(apca);
  }

  /*
   * Compute reconstruction error over intervall [pos, pos+1]
   */
  private <E> double calcReconstructionError(TimeSeries timeSeries, List<Integer> keys, List<Double> values, int pos) {
    double reconstructionError = 0.0;
    int start = (pos > 0)? keys.get(pos-1):0;
    int end = keys.get(pos);
    double mean = values.get(pos);
    for (int i = start; i < end; i++) {
      double value = (mean-timeSeries.getData()[i]);
      reconstructionError += value*value;
    }
    return reconstructionError;
  }

  /*
   * Compute reconstruction error over intervall [index, index+1]
   */
  private <E> double calcJointReconstructionError(TimeSeries timeSeries, List<Integer> keys, List<Double> values, int index) {
    double reconstructionError = 0.0;
    int startFirst = index>0? keys.get(index-1) : 0;
    int endFirst = keys.get(index);
    int endSecond = keys.get(index+1);

    double mean = ((endFirst-startFirst) * values.get(index) +  (endSecond-endFirst) * values.get(index+1)) / (double)(endSecond - startFirst);

    for (int i = startFirst; i < endSecond; i++) {
      double value = (mean - timeSeries.getData()[i]);
      reconstructionError += value*value;
    }
    return reconstructionError;
  }

  public TimeSeries inverseTransform(TimeSeries timeSeries, int n) {
    int l = timeSeries.getLength();
    double[] apca = timeSeries.getData();
    double[] data = new double[n];

    //  for every intervall, insert the mean
    int start = 0;
    for (int i = 0; i < l; i+=2) {
      for (int j = start; j < (int)apca[i+1]; j++) {
        data[j] = apca[i];
      }
      start = (int)apca[i+1];
    }

    return new TimeSeries(data);
  }

  public double getDistance(TimeSeries t1, TimeSeries query, int n, double minValue) {

    int l = t1.getLength();

    double distance = 0.0;
    double[] data = t1.getData();
    double[] data2 = query.getData();

    int offset1 = 1;
    int offset2 = 1;

    int start = 0;
    int end1 = 0;
    int end2 = 0;

    while(end1 < n || end2 <n) {
      end1 = (int)data[offset1];
      end2 = (int)data2[offset2];

      int end = Math.min(end1, end2);
      for (int j = start; j < end; j++) {
        // compute differences
        double value = data[offset1-1]-data2[offset2-1];
        distance += value*value;
      }

      start = end;
      if (start >= end1) {
        offset1+=2;
      }
      if (start >= end2) {
        offset2+=2;
      }

      // early stopping
      if (distance > minValue) {
        return Double.POSITIVE_INFINITY;
      }
    }
    return distance;
  }

  public double getDistance(TimeSeries t1, TimeSeries query, TimeSeries originalQuery, int n, double minValue) {

    int l = t1.getLength();

    double distance = 0;
    double[] data = t1.getData();
    double[] data2 = originalQuery.getData();//inverseTransform(query, n).getData();

    int start = 0;
    for (int q = 1; q < l; q+=2) {
      double end = data[q];

      double mean = 0.0;
      for (int i = start; i < end; i++) {
        mean += data2[i];
      }
      mean /= end - start;

      double value = (data[q-1] - mean);
      distance += value*value * (end-start);

      if (distance > minValue) {
        return Double.POSITIVE_INFINITY;
      }

      start = (int)end;
    }

    return distance;
  }
}
