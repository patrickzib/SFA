package sfa.transformation;


/**
 * SFA lower bounding distance of the Euclidean distance as published in
 *    Schäfer, P., Högqvist, M.: SFA: a symbolic fourier approximation and 
 *    index for similarity search in high dimensional datasets. 
 *    In: EDBT, ACM (2012) *
 */
public class SFADistance {
  private SFA sfa;
  
  public SFADistance(SFA transform) {
    this.sfa = transform;
  }
  
  /**
   * SFA-distance
   */
  public double getDistance(short[] wordsTs, short[] wordsQuery, double[] dftQuery, boolean normed, double minValue) {
    double distance = 0.0;
    
    // mean (DC) value
    int i = 0;
    if (!normed) {
      distance = dist(
          wordsTs[0],
          wordsQuery[0],
          dftQuery[0],
          0);
      distance *= distance;
      i+=2;
    }

    // remaining Fourier values (mean and imaginary parts)
    for (; i < wordsTs.length; i++) {

      double value = dist(
          wordsTs[i],
          wordsQuery[i],
          dftQuery[i],
          i);
          
      distance += 2*value*value;

      // pruning if distance threshold is exceeded
      if (distance > minValue) {
        return distance;
      }
    }

    return distance;
  }
  
  /**
   * Distance between a symbol and a Fourier value based on bins
   */
  public double dist (short c1Value, short c2Value, double realC2, int dim) {
    if (c1Value == c2Value) {
      return 0;
    }
    return (c1Value > c2Value)?
        ((SFA)this.sfa).bins[dim][c1Value-1] - realC2 :
          realC2 - ((SFA)this.sfa).bins[dim][c1Value];
  }
}
