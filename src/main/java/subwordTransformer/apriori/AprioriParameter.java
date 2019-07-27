package subwordTransformer.apriori;

import java.util.ArrayList;
import java.util.List;

import subwordTransformer.Parameter;

/**
 * The parameter for the AprioriTransformer.
 */
public class AprioriParameter extends Parameter {

  private final int MIN_SIZE;
  private final double MIN_SUPPORT;

  /**
   * @param minSize    the minimal number of characters that the patterns should
   *                   have
   * @param minSupport the minimal support for patterns that should be found by
   *                   the transformer
   */
  public AprioriParameter(int minSize, double minSupport) {
    this.MIN_SIZE = minSize;
    this.MIN_SUPPORT = minSupport;
  }

  /**
   * @return the minimal size for patterns
   */
  public int getMinSize() {
    return this.MIN_SIZE;
  }

  /**
   * @return the minimal support
   */
  public double getMinSupport() {
    return this.MIN_SUPPORT;
  }

  /**
   * @param minSupportStart the start of the minimal support
   * @param minSupportEnd   the end of the minimal support
   * @param supportStep     the increase of the minimal support
   * @return a list of AprioriParameters sorted ascendingly by minimal support
   */
  public static List<AprioriParameter> getParameterList(int minSizeStart, int minSizeEnd, double minSupportStart, double minSupportEnd, double supportStep) {
    List<AprioriParameter> list = new ArrayList<>();
    for (double minSupport = minSupportStart; minSupport <= minSupportEnd; minSupport += supportStep) {
      for (int minSize = minSizeStart; minSize <= minSizeEnd; minSize++) {
        list.add(new AprioriParameter(minSize, minSupport));
      }
    }
    return list;
  }

  @Override
  public String toString() {
    return "AprioriParameter(minSize=" + this.getMinSize() + ", minSupport=" + this.getMinSupport() + ")";
  }
}
