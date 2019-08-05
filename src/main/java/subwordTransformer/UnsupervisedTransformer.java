package subwordTransformer;

/**
 * Transforms words to subwords after unsupervised training on a given dataset.
 * 
 * @param <P> the parameter type that the transformer uses
 */
public abstract class UnsupervisedTransformer<P extends Parameter> extends SubwordTransformer<P> {

  private short[][] currentWords;

  /**
   * @param alphabetSize        the alphabet size of the input words
   * @param positionalAlphabets whether the transformer should use positional
   *                            alphabets, i.e. different positions in words have
   *                            different meanings
   * @param fillCharacter       the character to be used for wildcards (default:
   *                            -1)
   */
  public UnsupervisedTransformer(int alphabetSize, boolean positionalAlphabets, short fillCharacter) {
    super(alphabetSize, positionalAlphabets, fillCharacter);
  }

  /**
   * @param alphabetSize        the alphabet size of the input words
   * @param positionalAlphabets whether the transformer should use positional
   *                            alphabets, i.e. different positions in words have
   *                            different meanings
   */
  public UnsupervisedTransformer(int alphabetSize, boolean positionalAlphabets) {
    super(alphabetSize, positionalAlphabets);
  }

  @Override
  protected boolean hasWords() {
    return this.currentWords != null;
  }

  protected short[][] getWords() {
    return this.currentWords;
  }

  /**
   * Sets the training words.
   * 
   * @param words the training words
   */
  public void setWords(short[][] words) {
    if (this.currentWords != words) {
      this.setWordsChanged();
      this.currentWords = words;
    }
  }

}
