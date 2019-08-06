package subwordTransformer.cng;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import subwordTransformer.SupervisedTransformer;
import subwordTransformer.SupervisedUtils;

/**
 * A transformer that searches for frequent character-n-grams.
 */
public class SupervisedCNGTransformer extends SupervisedTransformer<CNGParameter> {

  private Map<Integer, List<Map<List<Short>, Integer>>> nGramCounts;
  private List<short[]> dictionary;

  /**
   * @param alphabetSize        the alphabet size of the input words
   * @param positionalAlphabets whether the transformer should use positional
   *                            alphabets, i.e. different positions in words have
   *                            different meanings
   */
  public SupervisedCNGTransformer(int alphabetSize, boolean positionalAlphabets) {
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
  public SupervisedCNGTransformer(int alphabetSize, boolean positionalAlphabets, short fillCharacter) {
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
    List<Map<List<Short>, Integer>> classCountList = new ArrayList<>();
    for (short[][] classWords : this.getWords()) {
      classCountList.add(CNGUtils.countNGrams(n, classWords, this.hasPositionalAlphabets(), this.getFillCharacter()));
    }
    nGramCounts.put(n, classCountList);
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
    int numClasses = this.getWords().length;
    int[] classSizes = new int[numClasses];
    for (int i = 0; i < numClasses; i++) {
      classSizes[i] = this.getWords()[i].length;
    }
    double max = Math.sqrt((numClasses - 1) / Math.pow(numClasses, 2));
    for (int n = this.getParameter().getMinN(); n <= this.getParameter().getMaxN(); n++) {
      Set<List<Short>> seenWords = new HashSet<>();
      List<Map<List<Short>, Integer>> classCounts = nGramCounts.get(n);
      for (int i = 0; i < numClasses; i++) {
        for (List<Short> word : classCounts.get(i).keySet()) {
          if (!seenWords.contains(word)) {
            seenWords.add(word);
            int[] counts = new int[numClasses];
            for (int j = 0; j < numClasses; j++) {
              Integer count = classCounts.get(j).get(word);
              counts[j] = count == null ? 0 : count.intValue();
            }
            if (SupervisedUtils.sigma(counts, classSizes, max) >= this.getParameter().getMinSupport()) {
              dictionary.add(CNGUtils.shortListToArray(word));
            }
          }
        }
      }
    }
  }

  @Override
  public short[][] transform(short[] word) {
    return CNGUtils.transform(word, dictionary, this.hasPositionalAlphabets(), this.getFillCharacter());
  }

}
