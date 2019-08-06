package subwordTransformer.cng;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import subwordTransformer.UnsupervisedTransformer;

/**
 * A transformer that searches for frequent character-n-grams.
 */
public class CNGTransformer extends UnsupervisedTransformer<CNGParameter> {

  private Map<Integer, Map<List<Short>, Integer>> nGramCounts;
  private List<short[]> dictionary;

  /**
   * @param alphabetSize        the alphabet size of the input words
   * @param positionalAlphabets whether the transformer should use positional
   *                            alphabets, i.e. different positions in words have
   *                            different meanings
   */
  public CNGTransformer(int alphabetSize, boolean positionalAlphabets) {
    super(alphabetSize, positionalAlphabets);
  }

  /**
   * @param alphabetSize        the alphabet size of the input words
   * @param positionalAlphabets whether the transformer should use positional
   *                            alphabets, i.e. different positions in words have
   *                            different meanings
   * @param fillCharacter       the character to be used for wildcards (default:
   *                            -1)
   */
  public CNGTransformer(int alphabetSize, boolean positionalAlphabets, short fillCharacter) {
    super(alphabetSize, positionalAlphabets, fillCharacter);
  }

  @Override
  public int getOutputAlphabetSize() {
    if (this.hasPositionalAlphabets()) {
      return this.getInputAlphabetSize() + 1;
    } else {
      return this.getInputAlphabetSize();
    }
  }

  private void countNGrams(int n) {
    nGramCounts.put(n, CNGUtils.countNGrams(n, this.getWords(), this.hasPositionalAlphabets(), this.getFillCharacter()));
  }

  @Override
  protected void buildDictionary() {
    nGramCounts = new HashMap<>();
    for (int n = this.getParameter().getMinN(); n <= this.getParameter().getMaxN(); n++) {
      this.countNGrams(n);
    }
    this.fillDictionary();
  }

  @Override
  protected void updateDictionary() {
    for (int n = this.getParameter().getMinN(); n <= this.getParameter().getMaxN(); n++) {
      if (!nGramCounts.containsKey(n)) {
        this.countNGrams(n);
      }
    }
    this.fillDictionary();
  }

  private void fillDictionary() {
    dictionary = new ArrayList<>();
    int minCount = (int) Math.ceil(this.getWords().length * this.getParameter().getMinSupport());
    for (int n = this.getParameter().getMinN(); n <= this.getParameter().getMaxN(); n++) {
      for (Map.Entry<List<Short>, Integer> subword : nGramCounts.get(n).entrySet()) {
        if (subword.getValue() >= minCount) {
          dictionary.add(CNGUtils.shortListToArray(subword.getKey()));
        }
      }
    }
  }

  @Override
  public short[][] transform(short[] word) {
    return CNGUtils.transform(word, dictionary, this.hasPositionalAlphabets(), this.getFillCharacter());
  }

}
