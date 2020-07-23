package sfa.transformation;

import java.io.Serializable;

import library.wavelets.lift.Haar;
import sfa.timeseries.TimeSeries;

/**
 * Haar-Wavelet Transform
 */
public class DWT extends Representation implements Serializable {
  private static final long serialVersionUID = -7092860432113787661L;

  final static transient Haar HAAR = new Haar();

  public DWT() {
    super();
  }

  /**
   * Discrete Wavelet Transform using Haar-Wavelets
   */
  public TimeSeries transform(TimeSeries timeSeries, int l) {
    // get next power of two
    int nextPowerOfTwo = nextPowerOfTwo(timeSeries.getLength());

    // Copy values to an array of length of a power of two
    // as Wavelet-Transform is in-place
    double[] values = new double[nextPowerOfTwo];
    System.arraycopy(timeSeries.getData(), 0, values, 0, timeSeries.getLength());

    HAAR.forwardTrans(values);

    // stores the first n coefficients
    double[] cutValues = new double[l];

    // take the fist n coefficients
    // norm the differences, such that lower bounding holds and Distanz D(x,y) = D(X,Y)
    for (int i = 0; i < Math.min(l, values.length); i++) {
      cutValues[i] = -0.5d*values[i];
    }
    //if (this.normMean) {
    //  cutValues[0] = values[0]; // remove overall mean value
    //}

    return new TimeSeries(cutValues);
  }


  public TimeSeries inverseTransform(TimeSeries timeSeries, int n) {

    // adapt to power of 2
    int nextPowerOfTwo = nextPowerOfTwo(n);

    double[] values = new double[nextPowerOfTwo];
    System.arraycopy(timeSeries.getData(), 0, values, 0, timeSeries.getLength());

    // norm the differences
    for (int i = 1; i < values.length; i++) {
      values[i] *= -2.0d;
    }
    HAAR.inverseTrans(values);

    double[] originalValues = new double[n];
    System.arraycopy(values, 0, originalValues, 0, n);
    return new TimeSeries (originalValues);
  }

  public double getDistance(TimeSeries t1, TimeSeries t2, TimeSeries originalQuery, int n, double minValue) {

    final int l = t1.getLength();

    double[] differences = new double[l];
    double[] t1Values = t1.getData();
    double[] t2Values = t2.getData();

    // compute differences
    for (int i = 0; i < l; i++) {
      differences[i] = t1Values[i] - t2Values[i];
      differences[i] = differences[i]*differences[i];
    }

    int log2n = (int)(Math.log(Representation.closestPowerOfTwo(n))/Math.log(2));
    return calcSiIterativ(differences, log2n, minValue);
  }

  protected double calcSiIterativ (double[] differences, int iplus1, double minValue) {

    double[] Si = new double[iplus1+1];
    Si[0] = differences[0];

    for (int i = 1; i <= iplus1; i++) {
      double result = Si[i-1]; // contains S(i-1)^2

      // from 2^(i-1) bis 2^(i)-1
      for (int j = (int)Math.pow(2,(i-1)); (j < (int)Math.pow(2,i)) && (j < differences.length); j++) {
        result += differences[j];

        if (result+result > minValue) {
          return Double.POSITIVE_INFINITY;
        }
      }
      Si[i] = 2*result;
    }

    return Si[iplus1];
  }
}
