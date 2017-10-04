// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import sfa.timeseries.TimeSeries;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * SFA using the ANOVA F-statistic to determine the best Fourier coefficients
 * (those that best separate between class labels) as opposed to using the first
 * ones.
 */
public class SFASupervised extends SFA {
  private static final long serialVersionUID = -6435016083374045799L;
  public int[] bestValues;

  public SFASupervised(HistogramType histType) {
    super(histType);
    this.lowerBounding = false;
  }

  public SFASupervised() {
    super(HistogramType.INFORMATION_GAIN);
    this.lowerBounding = false;
  }

  /**
   * Quantization of a DFT approximation to its SFA word
   *
   * @param approximation the DFT approximation of a time series
   * @return
   */
  @Override
  public short[] quantization(double[] approximation) {
    short[] signal = new short[Math.min(approximation.length, this.bestValues.length)];

    for (int a = 0; a < signal.length; a++) {
      int i = this.bestValues[a];
      // lookup character:
      short beta = 0;
      for (beta = 0; beta < this.bins[i].length; beta++) {
        if (approximation[i] < this.bins[i][beta]) {
          break;
        }
      }
      signal[a] = beta;
    }

    return signal;
  }

  /**
   * Trains the SFA boss based on a set of samples. At the end of this call,
   * the quantization bins are set.
   *
   * @param samples    the samples to use for training.
   * @param wordLength Length of the resulting SFA words. Each character of a word
   *                   corresponds to one Fourier value. As opposed to the normal SFA
   *                   boss, here characters correspond to those Fourier values that are
   *                   most distinctive between class labels.
   * @param symbols    the alphabet size, i.e. number of quantization bins to use
   * @param normMean   true: sets mean to 0 for each time series.
   * @return the Fourier transformation of the time series.
   */
  @Override
  public short[][] fitTransform(TimeSeries[] samples, int wordLength, int symbols, boolean normMean) {
    int length = samples[0].getLength();
    double[][] transformedSignal = fitTransformDouble(samples, length, symbols, normMean);

    Indices<Double>[] best = calcBestCoefficients(samples, transformedSignal);

    // use best coefficients (the ones with largest f-value)
    this.bestValues = new int[Math.min(best.length, wordLength)];
    this.maxWordLength = 0;
    for (int i = 0; i < this.bestValues.length; i++) {
      this.bestValues[i] = best[i].index;
      this.maxWordLength = Math.max(best[i].index + 1, this.maxWordLength);
    }

    // make sure it is an even number
    this.maxWordLength += this.maxWordLength % 2;

    return transform(samples, transformedSignal);
  }

  /**
   * calculate ANOVA F-stat
   * compare : https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/feature_selection/univariate_selection.py#L121
   *
   * @param transformedSignal
   * @return
   */
  public static Indices<Double>[] calcBestCoefficients(
      TimeSeries[] samples,
      double[][] transformedSignal) {
    HashMap<Double, ArrayList<double[]>> classes = new HashMap<>();
    for (int i = 0; i < samples.length; i++) {
      ArrayList<double[]> allTs = classes.get(samples[i].getLabel());
      if (allTs == null) {
        allTs = new ArrayList<>();
        classes.put(samples[i].getLabel(), allTs);
      }
      allTs.add(transformedSignal[i]);
    }

    double nSamples = transformedSignal.length;
    double nClasses = classes.keySet().size();
    int length = transformedSignal[0].length;

    double[] f = getFoneway(length, classes, nSamples, nClasses);

    // sort by largest f-value
    @SuppressWarnings("unchecked")
    Indices<Double>[] best = new Indices[f.length];
    for (int i = 0; i < f.length; i++) {
      best[i] = new Indices<>(i, f[i]);
    }
    Arrays.sort(best);
    return best;
  }

  /**
   * The one-way ANOVA tests the null hypothesis that 2 or more groups have
   * the same population mean. The test is applied to samples from two or
   * more groups, possibly with differing sizes.
   *
   * @param length
   * @param classes
   * @param nSamples
   * @param nClasses
   * @return
   */
  public static double[] getFoneway(
      int length,
      Map<Double, ArrayList<double[]>> classes,
      double nSamples,
      double nClasses) {
    double[] ss_alldata = new double[length];
    HashMap<Double, double[]> sums_args = new HashMap<>();

    for (Entry<Double, ArrayList<double[]>> allTs : classes.entrySet()) {

      double[] sums = new double[ss_alldata.length];
      sums_args.put(allTs.getKey(), sums);

      for (double[] ts : allTs.getValue()) {
        for (int i = 0; i < ts.length; i++) {
          ss_alldata[i] += ts[i] * ts[i];
          sums[i] += ts[i];
        }
      }
    }

    double[] square_of_sums_alldata = new double[ss_alldata.length];
    Map<Double, double[]> square_of_sums_args = new HashMap<>();
    for (Entry<Double, double[]> sums : sums_args.entrySet()) {
      for (int i = 0; i < sums.getValue().length; i++) {
        square_of_sums_alldata[i] += sums.getValue()[i];
      }

      double[] squares = new double[sums.getValue().length];
      square_of_sums_args.put(sums.getKey(), squares);
      for (int i = 0; i < sums.getValue().length; i++) {
        squares[i] += sums.getValue()[i] * sums.getValue()[i];
      }
    }

    for (int i = 0; i < square_of_sums_alldata.length; i++) {
      square_of_sums_alldata[i] *= square_of_sums_alldata[i];
    }

    double[] sstot = new double[ss_alldata.length];
    for (int i = 0; i < sstot.length; i++) {
      sstot[i] = ss_alldata[i] - square_of_sums_alldata[i] / nSamples;
    }

    double[] ssbn = new double[ss_alldata.length];    // sum of squares between
    double[] sswn = new double[ss_alldata.length];    // sum of squares within

    for (Entry<Double, double[]> sums : square_of_sums_args.entrySet()) {
      double n_samples_per_class = classes.get(sums.getKey()).size();
      for (int i = 0; i < sums.getValue().length; i++) {
        ssbn[i] += sums.getValue()[i] / n_samples_per_class;
      }
    }

    for (int i = 0; i < square_of_sums_alldata.length; i++) {
      ssbn[i] -= square_of_sums_alldata[i] / nSamples;
    }

    double dfbn = nClasses - 1;                       // degrees of freedom between
    double dfwn = nSamples - nClasses;              // degrees of freedom within
    double[] msb = new double[ss_alldata.length];   // variance (mean square) between classes
    double[] msw = new double[ss_alldata.length];   // variance (mean square) within samples
    double[] f = new double[ss_alldata.length];     // f-ratio

    for (int i = 0; i < sswn.length; i++) {
      sswn[i] = sstot[i] - ssbn[i];
      msb[i] = ssbn[i] / dfbn;
      msw[i] = sswn[i] / dfwn;
      f[i] = msb[i] / msw[i];
    }
    return f;
  }

  static class Indices<E extends Comparable<E>> implements Comparable<Indices<E>> {
    int index;
    E value;

    public Indices(int index, E value) {
      this.index = index;
      this.value = value;
    }

    public int compareTo(Indices<E> o) {
      return o.value.compareTo(this.value); // descending sort!
    }

    @Override
    public String toString() {
      return "(" + this.index + ":" + this.value + ")";
    }
  }
}
