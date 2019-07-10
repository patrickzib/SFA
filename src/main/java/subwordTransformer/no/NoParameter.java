package subwordTransformer.no;

import java.util.ArrayList;
import java.util.List;

import subwordTransformer.Parameter;

/**
 * The parameter for the NoTransformer.
 */
public class NoParameter extends Parameter {

  public NoParameter() {
  }

  /**
   * @return a list with a single NoParameter
   */
  public static List<NoParameter> getParameterList() {
    List<NoParameter> list = new ArrayList<>();
    list.add(new NoParameter());
    return list;
  }

  @Override
  public String toString() {
    return "NoParameter()";
  }
}
