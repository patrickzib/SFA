// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import sfa.timeseries.TimeSeries;

/**
 * A dimensionality reduction technique
 */
public abstract class Representation implements Serializable {
  public void init() {
  }

  /**
   * Transforms a time series using l coefficients
   * @param timeSeries
   * @param l number of coefficients in the target representation
   * @return
   */
  public abstract TimeSeries transform (TimeSeries timeSeries, int l);

  /**
   * Inverse transform from approximate representation
   * @param timeSeries
   * @param n target length
   * @return
   */
  public abstract TimeSeries inverseTransform (TimeSeries timeSeries, int n);

  /**
   * A lower bounding distance measure on the representation
   * @param t
   * @param query
   * @param n the original length before transformation
   * @param minValue distance for early stopping
   * @return
   */
  public abstract double getDistance(TimeSeries t, TimeSeries query, TimeSeries originalQuery, int n, double minValue);

  /**
   * Transforms a set of time series using n coefficients
   * @param samples
   * @param l
   * @return
   */
  public TimeSeries[] transform(TimeSeries[] samples, int l) {
    if (samples!= null) {
      TimeSeries[] transform = new TimeSeries[samples.length];
      for (int i = 0; i < transform.length; i++) {
        transform[i] = transform(samples[i], l);
      }
      return transform;
    }
    return null;
  }

  /**
   * get log2 exponent
   * @param n
   * @return
   */
  public static int closestPowerOfTwo (int n) {
    if (((n) & (n-1)) == 0) {
      return n;
    }
    return Integer.highestOneBit(n-1)<<1;
  }

  /**
   * get the next power of two
   * @param x
   */
  public static int nextPowerOfTwo (int x) {
    return closestPowerOfTwo(x);
  }

  public static Representation loadFromDisk(String path) {
    try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path));) {
      return (Representation) in.readObject();
    }
    catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  public boolean writeToDisk(String path) {
    try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
      out.writeObject(this);
      return true;
    } catch (IOException e) {
      e.printStackTrace();
    }
    return false;
  }

}
