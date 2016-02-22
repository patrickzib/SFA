// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import java.util.Arrays;

import sfa.timeseries.TimeSeries;

/**
 * The Momentary Fourier Transform is alternative algorithm of
 * the Discrete Fourier Transform for overlapping windows. It has
 * a constant computational complexity for in the window length n as 
 * opposed to O(n log n) for the Fast Fourier Transform algorithm. 
 * 
 * It was first published in:
 *    Albrecht, S., Cumming, I., Dudas, J.: The momentary fourier transformation 
 *    derived from recursive matrix transformations. In: Digital Signal Processing 
 *    Proceedings, 1997., IEEE (1997)
 *  
 * @author bzcschae
 *
 */
public class MFT {
  boolean normMean = false;

  public MFT(boolean normMean) {
    this.normMean = normMean;
  }

  public double[] transform(TimeSeries ts, int windowSize, int l) {
    return DFTNormed(ts.getData(), ts.calculateStddev(), windowSize, l);
  }
  
  public double[] DFTNormed(double[] data, double std, int windowSize, int l) {
    double[] dft = DFT(data, windowSize, l);
    return normalizeFT(dft, std, 1.0/Math.sqrt(windowSize));
  }
  
  public double[] DFT(double[] data, int windowSize, int l) {
    double[] dft = new double[l];
    double phi = 2*Math.PI / windowSize;
    int startOffset = normMean? 1 : 0;

    for (int k = startOffset; k < l/2+startOffset; k++) { 
      double real = 0.0;
      double imag = 0.0;
      for (int t = 0; t < windowSize; t++) { 
        real += data[t]*Math.cos(phi * t * k);
        imag += -data[t]*Math.sin(phi * t * k);
      }
      dft[(k-startOffset)*2] = real;
      dft[(k-startOffset)*2+1] = imag;
    }

    return dft;
  }

  public double[][] transformWindowing(TimeSeries timeSeries, int windowSize, int l) {
    // ignore DC value?
    int startOffset = normMean? 2 : 0;
    double norm = 1.0/Math.sqrt(windowSize);

    int wordLength = Math.min(windowSize-startOffset, l);    
    wordLength = wordLength + wordLength % 2; // make it even
    double[] phis = new double[wordLength];

    for (int u = 0; u < phis.length; u+=2) {
      double uHalve = -(u+startOffset)/2;
      phis[u] = realephi(uHalve, windowSize);
      phis[u+1] = complexephi(uHalve, windowSize);
    }

    // means and stddev for each sliding window
    int end = Math.max(1,timeSeries.getLength()-windowSize+1);
    double[] means = new double[end];
    double[] stds = new double[end];
    TimeSeries.calcIncreamentalMeanStddev(windowSize, timeSeries, means, stds);

    // holds the DFT of each sliding window
    double[][] transformed = new double[end][];
    double[] mftData = null;
    double[] data = timeSeries.getData();

    for (int t = 0; t < end; t++) {
      // use the MFT
      if (t > 0) {
        for (int k = 0; k < wordLength; k+=2) {
          double real1 = (mftData[k] + data[t+windowSize-1] - data[t-1]);
          double imag1 = (mftData[k+1]);

          double real = complexMulReal(real1, imag1, phis[k], phis[k+1]);
          double imag = complexMulImag(real1, imag1, phis[k], phis[k+1]);

          mftData[k] = real;
          mftData[k+1] = imag;
        }
      }
      // use the DFT for the first offset
      else {
        mftData = Arrays.copyOf(timeSeries.getData(), windowSize);
        mftData = DFT(mftData, windowSize, l);
      }

      // normalization for lower bounding
      transformed[t] = normalizeFT(Arrays.copyOf(mftData, l), stds[t], norm);
    }

    return transformed;
  }

  public static double complexMulReal(double r1, double im1, double r2, double im2) {
    return r1*r2 - im1*im2;
  }

  public static double complexMulImag(double r1, double im1, double r2, double im2) {
    return r1*im2 + r2*im1;
  }

  public static double realephi(double u, double M) {
    return Math.cos(2*Math.PI*u/M);
  }

  public static double complexephi(double u, double M) {
    return -Math.sin(2*Math.PI*u/M);
  }

  public double[] normalizeFT(double[] copy, double std, double norm) {
    double normalisingFactor = (std>0? 1.0 / std : 1.0) * norm;
    for (int i = 0; i < copy.length; i++) {
      copy[i] *= normalisingFactor;
    }
    return copy;
  }
}
