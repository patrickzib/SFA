package sfa.transformation;

import org.jtransforms.fft.DoubleFFT_1D;
import sfa.timeseries.TimeSeries;

public class DFT extends Representation {
  private static final long serialVersionUID = 864303984968234551L;

  public int startOffset = 2; // discard DC-coefficient
  private DoubleFFT_1D fft = null;
  private int localFFTSize = 0;

  public DFT() {
  }

  @Override
  public TimeSeries transform(TimeSeries timeSeries, int l) {
    if (l%2==1) {
      throw new IllegalArgumentException(
          "warning: l should be even to store real and imaginary parts.");
    }

    // power of 2 transformation
    int nextPowerOfTwo = this.fft != null? this.localFFTSize : nextPowerOfTwo(timeSeries.getLength());
    double[] dataCopy2 = new double[nextPowerOfTwo];
    System.arraycopy(timeSeries.getData(), 0, dataCopy2, 0, timeSeries.getLength());

    if (this.fft == null || this.localFFTSize != nextPowerOfTwo) {
      this.fft = new DoubleFFT_1D(nextPowerOfTwo);
      this.localFFTSize = nextPowerOfTwo;
    }

    this.fft.realForward(dataCopy2);

    dataCopy2[1] = 0; // sometimes there is a value > 0 here, through real valued input???
    double[] cutValues = new double[Math.min(l,nextPowerOfTwo-this.startOffset)];
    System.arraycopy(dataCopy2, this.startOffset, cutValues, 0, cutValues.length);

    // norming for lower bounding
    //if (lowerBoundingNorm) {
      double norm = 1.0/Math.sqrt(nextPowerOfTwo);
      for (int i = 0; i < cutValues.length; i++) {
        cutValues[i] *= norm;
        if (i%2==1) {
          cutValues[i] = -1*cutValues[i];
        }
      }
    //}

    return new TimeSeries(cutValues);
  }

  public TimeSeries inverseTransform(TimeSeries timeSeries, int n) {

    // next power of two
    int nextPowerOfTwo = nextPowerOfTwo(n);
    double[] values = new double[nextPowerOfTwo];
    int l = timeSeries.getLength();
    System.arraycopy(timeSeries.getData(), 0, values, 2, l);

    // remove normalization
    double norm = Math.sqrt(nextPowerOfTwo);
    for (int i = 0; i < values.length; i++) {
      values[i] *= norm;
      if (i%2==1) {
        values[i] = -1*values[i];
      }
    }

    this.fft.realInverse(values, false); // TODO scaling?
    return new TimeSeries (values);

  }

  protected static double magnitude (double real, double img) {
    return real*real + img*img;
  }

  @Override
  public double getDistance(TimeSeries t1, TimeSeries t2, TimeSeries originalQuery, int n, double minValue) {
    int t1Size = t1.getLength();
    double distance = 0;

    // in steps of 2 for real and img part
    // 2*|Xi-Qi|^2
    for (int i = 0; i < t1Size; i+=2) {
      distance += 2*magnitude(t1.getData()[i] - t2.getData()[i], t1.getData()[i+1] - t2.getData()[i+1]);

      // early stopping
      if (distance > minValue) {
        return Double.POSITIVE_INFINITY;
      }
    }
    return distance;
  }

}