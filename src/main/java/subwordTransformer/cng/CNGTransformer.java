package subwordTransformer.cng;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import subwordTransformer.SubwordTransformer;

/**
 * A transformer that searches for frequent character-n-grams.
 */
public class CNGTransformer extends SubwordTransformer<CNGParameter> {

  private final Map<Integer, Map<List<Short>, Integer>> nGramCounts = new HashMap<>();
  private final List<short[]> dictionary = new ArrayList<>();

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
    Map<List<Short>, Integer> countMap = new HashMap<>();
    for (short[] word : this.getWords()) {
      int lastIndex = word.length - n;
      for (int i = 0; i <= lastIndex; i++) {
        List<Short> subword;
        if (this.hasPositionalAlphabets()) {
          subword = shortArrayToListPattern(word, i, n, this.getFillCharacter());
        } else {
          subword = shortArrayToList(word, i, n);
        }
        if (countMap.containsKey(subword)) {
          countMap.put(subword, countMap.get(subword) + 1);
        } else {
          countMap.put(subword, 1);
        }
      }
    }
    nGramCounts.put(n, countMap);
  }

  @Override
  protected void buildDictionary() {
    nGramCounts.clear();
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
    dictionary.clear();
    int minCount = (int) Math.ceil(this.getWords().length * this.getParameter().getMinSupport());
    for (int n = this.getParameter().getMinN(); n <= this.getParameter().getMaxN(); n++) {
      for (Map.Entry<List<Short>, Integer> subword : nGramCounts.get(n).entrySet()) {
        if (subword.getValue() >= minCount) {
          dictionary.add(shortListToArray(subword.getKey()));
        }
      }
    }
  }

  @Override
  public short[][] transform(short[] word) {
    List<short[]> matchingSubwords = new ArrayList<>();
    for (short[] subword : dictionary) {
      if ((this.hasPositionalAlphabets() && matchesWord(subword, word, this.getFillCharacter())) || (!this.hasPositionalAlphabets() && isSubArray(subword, word))) {
        matchingSubwords.add(subword);
      }
    }
    return matchingSubwords.toArray(new short[matchingSubwords.size()][]);
  }

  private static List<Short> shortArrayToList(short[] word, int startIndex, int length) {
    List<Short> wordList = new ArrayList<>();
    for (int i = startIndex; i < startIndex + length; i++) {
      wordList.add(word[i]);
    }
    return wordList;
  }

  private static List<Short> shortArrayToListPattern(short[] word, int startIndex, int length, short wildcard) {
    List<Short> wordList = new ArrayList<>();
    for (int i = 0; i < startIndex; i++) {
      wordList.add(wildcard);
    }
    wordList.addAll(shortArrayToList(word, startIndex, length));
    for (int i = startIndex + length; i < word.length; i++) {
      wordList.add(wildcard);
    }
    return wordList;
  }

  private static short[] shortListToArray(List<Short> word) {
    short[] wordArray = new short[word.size()];
    for (int i = 0; i < word.size(); i++) {
      wordArray[i] = word.get(i);
    }
    return wordArray;
  }

  private static boolean isSubArray(short[] subArray, short[] array) {
    int subArrayLength = subArray.length;
    int arrayLength = array.length;
    boolean contained = false;
    for (int i = 0; i <= arrayLength - subArrayLength; i++) {
      boolean found = true;
      for (int j = 0; j < subArrayLength; j++) {
        if (subArray[j] != array[i + j]) {
          found = false;
          break;
        }
      }
      if (found) {
        contained = true;
        break;
      }
    }
    return contained;
  }

  private static boolean matchesWord(short[] pattern, short[] word, short wildcard) {
    if (pattern.length != word.length) {
      return false;
    }
    for (int i = 0; i < pattern.length; i++) {
      if (pattern[i] != wildcard && pattern[i] != word[i]) {
        return false;
      }
    }
    return true;
  }

}
