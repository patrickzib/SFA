package subwordTransformer;

import java.util.Arrays;

/**
 * Transforms words to subwords after training on a given dataset.
 * 
 * @param <P> the parameter type that the transformer uses
 */
public abstract class SubwordTransformer<P extends Parameter> implements Cloneable {

  private P currentParam;
  private P oldParam;
  private short[][] currentWords;
  private boolean wordsChanged = false;
  private boolean paramChanged = false;
  private final int inputAlphabetSize;
  private final boolean positionalAlphabets;
  private final short fillCharacter;
  private boolean isTrained = false;

  /**
   * @param alphabetSize        the alphabet size of the input words
   * @param positionalAlphabets whether the transformer should use positional
   *                            alphabets, i.e. different positions in words have
   *                            different meanings
   * @param fillCharacter       the character to be used for wildcards (default:
   *                            -1)
   */
  public SubwordTransformer(int alphabetSize, boolean positionalAlphabets, short fillCharacter) {
    this.inputAlphabetSize = alphabetSize;
    this.positionalAlphabets = positionalAlphabets;
    this.fillCharacter = fillCharacter;
  }

  /**
   * @param alphabetSize        the alphabet size of the input words
   * @param positionalAlphabets whether the transformer should use positional
   *                            alphabets, i.e. different positions in words have
   *                            different meanings
   */
  public SubwordTransformer(int alphabetSize, boolean positionalAlphabets) {
    this(alphabetSize, positionalAlphabets, (short) -1);
  }

  /**
   * @return the alphabet size of the input words
   */
  public int getInputAlphabetSize() {
    return this.inputAlphabetSize;
  }

  /**
   * @return the alphabet size of the output words
   */
  public abstract int getOutputAlphabetSize();

  /**
   * @return whether the transformer uses positional alphabets, i.e. different
   *         positions in words have different meanings
   */
  public boolean hasPositionalAlphabets() {
    return this.positionalAlphabets;
  }

  /**
   * @return the character to be used for wildcards
   */
  public short getFillCharacter() {
    return this.fillCharacter;
  }

  /**
   * @return the parameter that is currently used
   */
  public P getParameter() {
    return this.currentParam;
  }

  protected P getOldParameter() {
    return this.oldParam;
  }

  protected short[][] getWords() {
    return this.currentWords;
  }

  /**
   * Sets the training words. The given array will be copied and not modified.
   * 
   * @param words the training words
   */
  public void setWords(short[][] words) {
    if (!Arrays.deepEquals(this.currentWords, words)) {
      this.isTrained = false;
      this.wordsChanged = true;
      // copy words array
      int wordCount = words.length;
      this.currentWords = new short[wordCount][];
      for (int i = 0; i < wordCount; i++) {
        short[] word = words[i];
        int wordLength = word.length;
        this.currentWords[i] = new short[wordLength];
        System.arraycopy(word, 0, this.currentWords[i], 0, wordLength);
      }
    }
  }

  /**
   * Sets the training parameter.
   * 
   * @param param the parameter
   */
  public void setParameter(P param) {
    if (param != this.currentParam) {
      this.isTrained = false;
      if (!this.paramChanged) {
        this.oldParam = this.currentParam;
        this.paramChanged = true;
      }
      this.currentParam = param;
    }
  }

  /**
   * Trains the transformer with the given parameter and training words.
   */
  public void fit() {
    if (this.currentWords == null) {
      throw new IllegalStateException("Requires training words to be set.");
    }
    if (this.currentParam == null) {
      throw new IllegalStateException("Requires the parameter to be set.");
    }
    this.isTrained = false;
    if (this.wordsChanged) {
      this.buildDictionary();
    } else if (this.paramChanged) {
      this.updateDictionary();
    }
    this.wordsChanged = false;
    this.paramChanged = false;
    this.isTrained = true;
  }

  protected abstract void buildDictionary();

  protected abstract void updateDictionary();

  /**
   * Transforms a word by matching it with the trained subwords.
   * 
   * @param word the word to be transformed
   * @return an array of matching subwords
   */
  public short[][] transformWord(short[] word) {
    if (!this.isTrained) {
      throw new IllegalStateException("Requires the transformer to be trained.");
    }
    return this.transform(word);
  }

  protected abstract short[][] transform(short[] word);

  @SuppressWarnings("unchecked")
  @Override
  public SubwordTransformer<P> clone() {
    try {
      return (SubwordTransformer<P>) super.clone();
    } catch (CloneNotSupportedException e) {
      throw new InternalError();
    }
  }

}
