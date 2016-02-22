package sfa.transformation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import sfa.timeseries.TimeSeries;

/**
 * Symbolic Fourier Approximation as published in
 *    Schäfer, P., Högqvist, M.: SFA: a symbolic fourier approximation and 
 *    index for similarity search in high dimensional datasets. 
 *    In: EDBT, ACM (2012)
 */
public class SFA {
  // distribution of values
  public ArrayList<ValueLabel>[] orderLine;

  public HistogramType histogramType = HistogramType.EQUI_DEPTH;

  public int alphabetSize = 256;
  public int wordLength = 0;
  public boolean initialized = false;

  public MFT transformation;
  
  // use binning / bucketing
  public double[][] bins;

  public enum HistogramType {
    EQUI_FREQUENCY, EQUI_DEPTH
  }

  static class ValueLabel {
    public double value;
    public String label;
    public ValueLabel(double key, String  label) {
      this.value = key;
      this.label = label;
    }
    @Override
    public String toString() {
      return "" + this.value + ":" + this.label;
    }
  }

  public SFA(HistogramType historgramType, boolean normMean) {
    reset();
    this.transformation = new MFT(normMean);
    this.histogramType = historgramType;
  }
 
  public void reset() {
    this.initialized = false;
    this.orderLine = null;
    this.bins = null;
  }

  @SuppressWarnings("unchecked")
  public void init(int l, int alphabetSize) {
    this.wordLength = l;
    this.alphabetSize = alphabetSize;
    this.initialized = true;

    // l-dimensional bins    
    this.alphabetSize = alphabetSize;

    this.bins = new double[l][alphabetSize-1];
    for (double[] row : this.bins ) {
      Arrays.fill(row, Double.MAX_VALUE);
    }

    this.orderLine = new ArrayList[l];
    for (int i = 0; i < this.orderLine.length; i++) {
      this.orderLine[i] = new ArrayList<ValueLabel>();
    }
  }

  public short[] transform(TimeSeries timeSeries) {
    return transform(timeSeries, null);
  }

  public short[] transform(TimeSeries timeSeries, double[] approximation) {
    if (!initialized) {
      throw new RuntimeException("Plase call fitTransform() first.");
    }
    if (approximation == null) {
      // get approximation of the time series
      approximation = this.transformation.transform(timeSeries, timeSeries.getLength(), this.wordLength);
    }

    // use lookup table (bins) to get the word from the approximation
    return quantization(approximation);
  }

  public short[][] transform(TimeSeries[] samples) {
    if (!initialized) {
      throw new RuntimeException("Plase call fitTransform() first.");
    }
    short[][] transform = new short[samples.length][];
    for (int i = 0; i < transform.length; i++) {
      transform[i] = transform(samples[i], null);
    }

    return transform;
  }
  
  public short[][] transform(TimeSeries[] samples, double[][] approximation) {
    if (!initialized) {
      throw new RuntimeException("Plase call fitTransform() first.");
    }    
    short[][] transform = new short[samples.length][];
    for (int i = 0; i < transform.length; i++) {
      transform[i] = transform(samples[i], approximation[i]);
    }

    return transform;
  }

  public short[] quantization(double[] approximation) {
    int i = 0;
    short[] word = new short[approximation.length];
    for (double value : approximation) {
      // lookup character:
      short c = 0;
      for (c = 0; c < this.bins[i].length; c++) {
        if (value < this.bins[i][c]) {
          break;
        }
      }
      word[i++] = c;
    }
    return word;
  }

  protected void sortOrderLine() {
    for (List<ValueLabel> element : this.orderLine) {
      Collections.sort(element, new Comparator<ValueLabel>() {
        @Override
        public int compare(ValueLabel o1, ValueLabel o2) {
          int comp = Double.compare(o1.value, o2.value);
          if (comp != 0) {
            return comp;
          }
          return o1.label != null ? o1.label.compareTo(o2.label) : 0;
        }
      });
    }
  }

  public void fitWindowing(TimeSeries[] timeSeries, int windowLength, int l, int symbols, boolean normMean) {
    this.transformation = new MFT(normMean);

    ArrayList<TimeSeries> sa = new ArrayList<TimeSeries>(timeSeries.length * timeSeries[0].getLength() / windowLength);
    for (TimeSeries t : timeSeries) {
      sa.addAll(Arrays.asList(t.getDisjointSequences(windowLength, normMean)));
    }
    fitTransform(sa.toArray(new TimeSeries[]{}), l, symbols);
  }

  public short[][] transformWindowing(TimeSeries ts, int windowLength, int l) {
    double[][] mft = this.transformation.transformWindowing(ts, windowLength, l);
    
    short[][] words = new short[mft.length][];
    for (int i = 0; i < mft.length; i++) {
//      if (mft[i].length != l) {
//        mft[i] = Arrays.copyOfRange(mft[i], 0, l);
//      }
      words[i] = quantization(mft[i]);      
    }
    
    return words;
  }

  public short[][] fitTransform (TimeSeries[] samples, int l, int symbols) {
    if (!this.initialized) {
      init(l, symbols);
    }

    double[][] transformedSamples = fillOrderline(samples, l, symbols);

    if (this.histogramType == HistogramType.EQUI_DEPTH) {
      divideEquiDepthHistogram();
    } else if (this.histogramType == HistogramType.EQUI_FREQUENCY) {
      divideEquiWidthHistogram();
    }
   
    return transform(samples, transformedSamples);
  }

  /**
   * Fill data in the orderline
   * @param samples
   */
  protected double[][] fillOrderline (TimeSeries[] samples, int l, int symbols) {
    double[][] transformedSamples = new double[samples.length][];

    for (int i = 0; i < samples.length; i++) {
      // z-normalization
      samples[i].norm();

      // approximation
      transformedSamples[i] = this.transformation.transform(samples[i], samples[i].getLength(), l);

      for (int j=0; j < transformedSamples[i].length; j++) {
        // round to 2 decimal places to reduce noise
        double value = Math.round(transformedSamples[i][j]*100.0)/100.0;
        this.orderLine[j].add(new ValueLabel(value, samples[i].getLabel()));
      }
    }

    // Sort ascending by value
    sortOrderLine();
    
    return transformedSamples;
  }

  protected void divideEquiWidthHistogram () {
    int i = 0;
    for (List<ValueLabel> elements : this.orderLine) {
      if (!elements.isEmpty()) {
        // apply the split
        double first = elements.get(0).value;
        double last = elements.get(elements.size()-1).value;
        double intervalWidth = (last-first) / (double)(this.alphabetSize);

        for (int c = 0; c < this.alphabetSize-1; c++) {
          this.bins[i][c] = intervalWidth*(c+1)+first;
        }
      }
      i++;
    }
  }

  protected void divideEquiDepthHistogram () {
    // For each real and imaginary part
    for (int i = 0; i < this.bins.length; i++) {
      // Divide into equi-depth intevals
      double depth = this.orderLine[i].size() / (double)(this.alphabetSize);

      int pos = 0;
      long count = 0;
      for (ValueLabel value : this.orderLine[i]) {
        if (++count > Math.ceil(depth*(pos+1))
            && (pos==0 || this.bins[i][pos-1] != value.value)) {
          this.bins[i][pos++] = value.value;
        }
      }
    }
  }

  public void printBins () {
    System.out.print("[");
    for (double[] element : this.bins) {
      System.out.print("-Inf\t" );
      for (double element2 : element) {
        String e = element2 != Double.MAX_VALUE? (""+element2) : "Inf";
        System.out.print("," + e + "\t");
      }
      System.out.println(";");
    }
    System.out.println("]");
  }
}
