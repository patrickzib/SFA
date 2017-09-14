// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import java.io.IOException;
import java.io.Serializable;

import org.jtransforms.fft.DoubleFFT_1D;

import sfa.timeseries.TimeSeries;

/**
 * The Momentary Fourier Transform is alternative algorithm of
 * the Discrete Fourier Transform for overlapping windows. It has
 * a constant computational complexity for in the window queryLength n as
 * opposed to O(n log n) for the Fast Fourier Transform algorithm.
 * <p>
 * It was first published in:
 * Albrecht, S., Cumming, I., Dudas, J.: The momentary fourier transformation
 * derived from recursive matrix transformations. In: Digital Signal Processing
 * Proceedings, 1997., IEEE (1997)
 *
 * @author bzcschae
 */
public class MFT implements Serializable {
  private static final long serialVersionUID = 8508604292241736378L;

  private int windowSize = 0;
  private int startOffset = 0;
  private double norm = 0;

  private transient DoubleFFT_1D fft = null;

  public MFT(int windowSize, boolean normMean, boolean lowerBounding) {
    this.windowSize = windowSize;
    this.fft = new DoubleFFT_1D(windowSize);

    // ignore DC value?
    this.startOffset = normMean ? 2 : 0;
    this.norm = lowerBounding ? 1.0 / Math.sqrt(windowSize) : 1.0;
  }

  /**
   * Transforms a time series using the *discrete* fourier transform. Results in
   * a single Fourier transform of the time series.
   *
   * @param timeSeries the time series to be transformed
   * @param l the number of Fourier values to keep
   * @return the first l Fourier values
   */
  public double[] transform(TimeSeries timeSeries, int l) {
    double[] data = new double[this.windowSize];
    int windowSize = timeSeries.getLength();

    System.arraycopy(timeSeries.getData(), 0, data, 0, this.windowSize);
    this.fft.realForward(data);
    data[1] = 0; // DC-coefficient imag part

    // norming
    double[] copy = new double[Math.min(windowSize - this.startOffset, l)];
    System.arraycopy(data, this.startOffset, copy, 0, copy.length);

    int sign = 1;
    for (int i = 0; i < copy.length; i++) {
      copy[i] *= this.norm * sign;
      sign *= -1;
    }

    return copy;
  }

  /**
   * Transforms a time series, extracting windows and using *momentary* fourier
   * transform for each window. Results in one Fourier transform for each
   * window. Returns only the first l/2 Fourier coefficients for each window.
   *
   * @param timeSeries the time series to be transformed
   * @param l          the number of Fourier values to use (equal to l/2 Fourier
   *                   coefficients). If l is uneven, l+1 Fourier values are returned. If
   *                   windowSize is smaller than l, only the first windowSize Fourier
   *                   values are set.
   * @return           returns only the first l/2 Fourier coefficients for each window.
   */
  public double[][] transformWindowing(TimeSeries timeSeries, int l) {
    int wordLength = l + l % 2 + this.startOffset; // make it even
    double[] phis = new double[wordLength];

    for (int u = 0; u < phis.length; u += 2) {
      double uHalve = -u / 2;
      phis[u] = realephi(uHalve, this.windowSize);
      phis[u + 1] = complexephi(uHalve, this.windowSize);
    }

    // means and stddev for each sliding window
    int end = Math.max(1, timeSeries.getLength() - this.windowSize + 1);
    double[] means = new double[end];
    double[] stds = new double[end];
    TimeSeries.calcIncrementalMeanStddev(this.windowSize, timeSeries.getData(), means, stds);

    double[][] transformed = new double[end][];

    // holds the DFT of each sliding window
    double[] mftData = new double[wordLength];
    double[] data = timeSeries.getData();

    for (int t = 0; t < end; t++) {
      // use the MFT
      if (t > 0) {
        for (int k = 0; k < wordLength; k += 2) {
          double real1 = (mftData[k] + data[t + this.windowSize - 1] - data[t - 1]);
          double imag1 = (mftData[k + 1]);

          double real = complexMulReal(real1, imag1, phis[k], phis[k + 1]);
          double imag = complexMulImag(real1, imag1, phis[k], phis[k + 1]);

          mftData[k] = real;
          mftData[k + 1] = imag;
        }
      }
      // use the DFT for the first offset
      else {
        double[] dft = new double[this.windowSize];
        System.arraycopy(timeSeries.getData(), 0, dft, 0, this.windowSize);

        this.fft.realForward(dft);
        dft[1] = 0; // DC-coefficient imag part

        // if windowSize > mftData.queryLength, the remaining data should be 0 now.
        System.arraycopy(dft, 0, mftData, 0, Math.min(mftData.length, dft.length));
      }

      // normalization for lower bounding
      double[] copy = new double[l];
      System.arraycopy(mftData, this.startOffset, copy, 0, l);

      transformed[t] = normalizeFT(copy, stds[t]);
    }

    return transformed;
  }

  public static double complexMulReal(double r1, double im1, double r2, double im2) {
    return r1 * r2 - im1 * im2;
  }

  public static double complexMulImag(double r1, double im1, double r2, double im2) {
    return r1 * im2 + r2 * im1;
  }

  public static double realephi(double u, double M) {
    return Math.cos(2 * Math.PI * u / M);
  }

  public static double complexephi(double u, double M) {
    return -Math.sin(2 * Math.PI * u / M);
  }

  public double[] normalizeFT(double[] copy, double std) {
    double normalisingFactor = (std > 0 ? 1.0 / std : 1.0) * this.norm;
    int sign = 1;
    for (int i = 0; i < copy.length; i++) {
      copy[i] *= sign * normalisingFactor;
      sign *= -1;
    }
    return copy;
  }

  private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
    in.defaultReadObject();
    this.fft = new DoubleFFT_1D(this.windowSize);
  }

}
