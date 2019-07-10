package subwordTransformer.cng;

import java.util.ArrayList;
import java.util.List;

import subwordTransformer.Parameter;

/**
 * The parameter for the CNGTransformer.
 */
public class CNGParameter extends Parameter {

  private final int MIN_N;
  private final int MAX_N;
  private final double MIN_SUPPORT;

  /**
   * @param minN       minimum length of character-n-grams
   * @param maxN       maximum length of character-n-grams
   * @param minSupport minimum support (frequency) of the character-n-grams
   */
  public CNGParameter(int minN, int maxN, double minSupport) {
    this.MIN_N = minN;
    this.MAX_N = maxN;
    this.MIN_SUPPORT = minSupport;
  }

  /**
   * @return minimum length of character-n-grams
   */
  public int getMinN() {
    return this.MIN_N;
  }

  /**
   * @return maximum length of character-n-grams
   */
  public int getMaxN() {
    return this.MAX_N;
  }

  /**
   * @return minimum support (frequency) of the character-n-grams
   */
  public double getMinSupport() {
    return this.MIN_SUPPORT;
  }

  /**
   * @param minNStart       the start of the minimum length
   * @param minNEnd         the end of the minimum length
   * @param maxNStart       the start of the maximum length
   * @param maxNEnd         the end of the maximum length
   * @param minSupportStart the start of the minimum support
   * @param minSupportEnd   the end of the minimum support
   * @param supportStep     the step of the minimum support
   * @return a list of CNGParameters with every possible combination of
   *         minimum/maximum word length and support
   */
  public static List<CNGParameter> getParameterList(int minNStart, int minNEnd, int maxNStart, int maxNEnd, double minSupportStart, double minSupportEnd, double supportStep) {
    List<CNGParameter> list = new ArrayList<>();
    for (int minN = minNStart; minN <= minNEnd; minN++) {
      for (int maxN = Math.max(minN, maxNStart); maxN <= maxNEnd; maxN++) {
        for (double minSupport = minSupportStart; minSupport <= minSupportEnd; minSupport += supportStep) {
          list.add(new CNGParameter(minN, maxN, minSupport));
        }
      }
    }
    return list;
  }

  @Override
  public String toString() {
    return "CNGParameter(n=" + this.getMinN() + ".." + this.getMaxN() + ", support=" + this.getMinSupport() + ")";
  }
}
