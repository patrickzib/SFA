package subwordTransformer.bpe;

import java.util.ArrayList;
import java.util.List;

import subwordTransformer.Parameter;

/**
 * The parameter for the BPETransformer.
 */
public class BPEParameter extends Parameter {

  private final double MIN_SUPPORT;

  /**
   * @param minSupport minimum support (frequency) of the merged characters
   */
  public BPEParameter(double minSupport) {
    this.MIN_SUPPORT = minSupport;
  }

  /**
   * @return minimum support (frequency) of the merged characters
   */
  public double getMinSupport() {
    return this.MIN_SUPPORT;
  }

  /**
   * @param minSupportStart the start of the minimum support
   * @param minSupportEnd   the end of the minimum support
   * @param supportStep     the step of the minimum support
   * @return a list of BPEParameters sorted descendingly by minimal support
   */
  public static List<BPEParameter> getParameterList(double minSupportStart, double minSupportEnd, double supportStep) {
    List<BPEParameter> list = new ArrayList<>();
    for (double minSupport = minSupportEnd; minSupport >= minSupportStart; minSupport -= supportStep) {
      list.add(new BPEParameter(minSupport));
    }
    return list;
  }

  @Override
  public String toString() {
    return "BPEParameter(support=" + this.getMinSupport() + ")";
  }
}
