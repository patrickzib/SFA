// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.serializers.FieldSerializer;
import org.jtransforms.fft.DoubleFFT_1D;
import sfa.timeseries.TimeSeries;

import java.io.IOException;
import java.io.Serializable;

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
 */
public class MFT implements Serializable {
  private static final long serialVersionUID = 8508604292241736378L;

  private int windowSize = 0;
  private int startOffset = 0;
  private double norm = 0;
  private boolean normMean = false;

  private transient DoubleFFT_1D fft = null;
  private boolean useMaxOrMin = false;

  public MFT() {
  }

  public MFT(int windowSize, boolean normMean, boolean lowerBounding) {
    this(windowSize, normMean, lowerBounding, false);
  }

  public MFT(int windowSize, boolean normMean, boolean lowerBounding, boolean useMinOrMax) {
    this.windowSize = windowSize;
    this.useMaxOrMin = useMinOrMax;

    initFFT();

    // ignore DC value?
    this.startOffset = normMean ? 2 : 0;
    this.norm = lowerBounding ? 1.0 / Math.sqrt(windowSize) : 1.0;
    this.normMean = normMean;
  }

  /**
   * Transforms a time series using the *discrete* fourier transform. Results in
   * a single Fourier transform of the time series.
   *
   * @param timeSeries the time series to be transformed
   * @param l          the number of Fourier values to keep
   * @return the first l Fourier values
   */
  public double[] transform(TimeSeries timeSeries, int l) {
    //if (!timeSeries.isNormed()) { // FIXME needed???
    //  timeSeries.norm(this.normMean);
    //}

    double[] data = new double[this.windowSize];
    System.arraycopy(timeSeries.getData(), 0, data, 0, Math.min(this.windowSize, timeSeries.getLength()));
    this.fft.realForward(data);
    data[1] = 0; // DC-coefficient imaginary part

    // make it even length for uneven windowSize
    double[] copy = new double[l];
    int length = Math.min(this.windowSize - this.startOffset, l);
    System.arraycopy(data, this.startOffset, copy, 0, length);

    // norming
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
   * @return returns only the first l/2 Fourier coefficients for each window.
   */
  public double[][] transformWindowing(TimeSeries timeSeries, int l) {
    int wordLength = useMaxOrMin ?
        Math.max(windowSize, l + this.startOffset) : // MUSE uses 'max'
        Math.min(windowSize, l + this.startOffset); // WEASEL uses 'min'
    wordLength += wordLength%2; // make it even
    double[] phis = new double[wordLength];

    for (int u = 0; u < phis.length; u += 2) {
      double uHalve = -u / 2;
      phis[u] = realPartEPhi(uHalve, this.windowSize);
      phis[u + 1] = complexPartEPhi(uHalve, this.windowSize);
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

          double real = complexMultiplyRealPart(real1, imag1, phis[k], phis[k + 1]);
          double imag = complexMultiplyImagPart(real1, imag1, phis[k], phis[k + 1]);

          mftData[k] = real;
          mftData[k + 1] = imag;
        }
      }
      // use the DFT for the first offset
      else {
        double[] dft = new double[this.windowSize];
        System.arraycopy(timeSeries.getData(), 0, dft, 0, Math.min(this.windowSize, timeSeries.getLength()));

        this.fft.realForward(dft);
        dft[1] = 0; // DC-coefficient imag part

        // if windowSize > mftData.queryLength, the remaining data should be 0 now.
        System.arraycopy(dft, 0, mftData, 0, Math.min(mftData.length, dft.length));
      }

      // normalization for lower bounding
      double[] copy = new double[l];
      System.arraycopy(mftData, this.startOffset, copy, 0, Math.min(l, mftData.length-this.startOffset));

      transformed[t] = normalizeFT(copy, stds[t]);
    }

    return transformed;
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
   * @return returns only the first l/2 Fourier coefficients for each window.
   */
  public short[][] transformWindowingShort(TimeSeries timeSeries, int l, SFA sfa) {
    int wordLength = useMaxOrMin ?
        Math.max(windowSize, l + this.startOffset) : // MUSE uses 'max'
        Math.min(windowSize, l + this.startOffset); // WEASEL uses 'min'
    wordLength += wordLength%2; // make it even
    double[] phis = new double[wordLength];

    for (int u = 0; u < phis.length; u += 2) {
      double uHalve = -u / 2;
      phis[u] = realPartEPhi(uHalve, this.windowSize);
      phis[u + 1] = complexPartEPhi(uHalve, this.windowSize);
    }

    // means and stddev for each sliding window
    int end = Math.max(1, timeSeries.getLength() - this.windowSize + 1);
    double[] means = new double[end];
    double[] stds = new double[end];
    TimeSeries.calcIncrementalMeanStddev(this.windowSize, timeSeries.getData(), means, stds);

    short[][] transformed = new short[end][];

    // holds the DFT of each sliding window
    double[] mftData = new double[wordLength];
    double[] data = timeSeries.getData();

    double[] copy = new double[l];

    for (int t = 0; t < end; t++) {
      // use the MFT
      if (t > 0) {
        for (int k = 0; k < wordLength; k += 2) {
          double real1 = (mftData[k] + data[t + this.windowSize - 1] - data[t - 1]);
          double imag1 = (mftData[k + 1]);

          double real = complexMultiplyRealPart(real1, imag1, phis[k], phis[k + 1]);
          double imag = complexMultiplyImagPart(real1, imag1, phis[k], phis[k + 1]);

          mftData[k] = real;
          mftData[k + 1] = imag;
        }
      }
      // use the DFT for the first offset
      else {
        double[] dft = new double[this.windowSize];
        System.arraycopy(timeSeries.getData(), 0, dft, 0, Math.min(this.windowSize, timeSeries.getLength()));

        this.fft.realForward(dft);
        dft[1] = 0; // DC-coefficient imag part

        // if windowSize > mftData.queryLength, the remaining data should be 0 now.
        System.arraycopy(dft, 0, mftData, 0, Math.min(mftData.length, dft.length));
      }

      // normalization for lower bounding
      System.arraycopy(mftData, this.startOffset, copy, 0, Math.min(l, mftData.length-this.startOffset));
      transformed[t] = sfa.quantization(normalizeFT(copy, stds[t]));
    }

    return transformed;
  }

  /**
   * Calculate the real part of a multiplication of two complex numbers
   */
  private static double complexMultiplyRealPart(double r1, double im1, double r2, double im2) {
    return r1 * r2 - im1 * im2;
  }

  /**
   * Caluculate the imaginary part of a multiplication of two complex numbers
   */
  private static double complexMultiplyImagPart(double r1, double im1, double r2, double im2) {
    return r1 * im2 + r2 * im1;
  }

  /**
   * Real part of e^(2*pi*u/M)
   */
  private static double realPartEPhi(double u, double M) {
    return Math.cos(2 * Math.PI * u / M);
  }

  /**
   * Imaginary part of e^(2*pi*u/M)
   */
  private static double complexPartEPhi(double u, double M) {
    return -Math.sin(2 * Math.PI * u / M);
  }

  /**
   * Apply normalization to the Fourier coefficients to allow lower bounding in Euclidean space
   */
  private double[] normalizeFT(double[] copy, double std) {
    double normalisingFactor = (TimeSeries.APPLY_Z_NORM && std > 0 ? 1.0 / std : 1.0) * this.norm;
    int sign = 1;
    for (int i = 0; i < copy.length; i++) {
      copy[i] *= sign * normalisingFactor;
      sign *= -1;
    }
    return copy;
  }

  public int getStartOffset() {
    return startOffset;
  }

  private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
    in.defaultReadObject();
    initFFT();
  }

  public static final class MFTKryoSerializer extends FieldSerializer<MFT> {

    public MFTKryoSerializer(Kryo kryo) {
      this(kryo, MFT.class);
    }

    public MFTKryoSerializer(Kryo kryo, Class type) {
      super(kryo, type);
    }

    @Override
    public MFT read(Kryo kryo, Input input, Class<MFT> type) {
      MFT mft = super.read(kryo, input, type);
      mft.initFFT();
      return mft;
    }
  }

  private void initFFT() {
    this.fft = new DoubleFFT_1D(this.windowSize);
  }
}
