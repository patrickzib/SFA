package subwordTransformer.no;

import subwordTransformer.UnsupervisedTransformer;

/**
 * A transformer that does not transform words (for testing purposes).
 */
public class NoTransformer extends UnsupervisedTransformer<NoParameter> {

  /**
   * @param alphabetSize the alphabet size of the input words
   */
  public NoTransformer(int alphabetSize) {
    super(alphabetSize, false);
  }

  @Override
  public int getOutputAlphabetSize() {
    return this.getInputAlphabetSize();
  }

  @Override
  protected void buildDictionary() {
  }

  @Override
  protected void updateDictionary() {
  }

  @Override
  public short[][] transform(short[] word) {
    return new short[][] {};
  }

}
