package subwordTransformer.bpe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import subwordTransformer.UnsupervisedTransformer;

/**
 * A transformer that uses byte pair encoding to find long representative
 * subsequences.
 */
public class BPETransformer extends UnsupervisedTransformer<BPEParameter> {

  private Map<List<List<Short>>, Integer> vocab;
  private List<List<List<Short>>> merges;

  private List<List<Short>> bestPair;
  private int bestCount;

  /**
   * @param alphabetSize        the alphabet size of the input words
   * @param positionalAlphabets whether the transformer should use positional
   *                            alphabets, i.e. different positions in words have
   *                            different meanings
   */
  public BPETransformer(int alphabetSize, boolean positionalAlphabets) {
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
  public BPETransformer(int alphabetSize, boolean positionalAlphabets, short fillCharacter) {
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

  @Override
  protected void buildDictionary() {
    // build vocab from words
    vocab = new HashMap<>();
    for (short[] word : this.getWords()) {
      List<List<Short>> wordList = this.getWordList(word);
      if (vocab.containsKey(wordList)) {
        vocab.put(wordList, vocab.get(wordList) + 1);
      } else {
        vocab.put(wordList, 1);
      }
    }
    // now find merges
    merges = new ArrayList<>();
    this.findMerges(false);
  }

  @Override
  protected void updateDictionary() {
    if (this.getParameter().getMinSupport() <= this.getOldParameter().getMinSupport()) {
      this.findMerges(true);
    } else {
      this.buildDictionary();
    }
  }

  private void findMerges(boolean continueSearch) {
    int minCount = (int) Math.ceil(this.getWords().length * this.getParameter().getMinSupport());
    while (true) {
      if (!continueSearch) {
        Map<List<List<Short>>, Integer> pairs = this.getStats();
        // find most frequent pair
        bestPair = null;
        bestCount = -1;
        for (Entry<List<List<Short>>, Integer> e : pairs.entrySet()) {
          if (e.getValue() > bestCount) {
            bestPair = e.getKey();
            bestCount = e.getValue();
          }
        }
      } else {
        continueSearch = false;
      }
      if (bestCount >= minCount) {
        merges.add(bestPair);
        this.mergeVocab(bestPair);
      } else {
        break;
      }
    }
  }

  private Map<List<List<Short>>, Integer> getStats() {
    Map<List<List<Short>>, Integer> pairs = new HashMap<>();
    for (Entry<List<List<Short>>, Integer> wordEntry : this.vocab.entrySet()) {
      List<List<Short>> word = wordEntry.getKey();
      for (int i = 0; i < word.size() - 1; i++) {
        List<List<Short>> pairList = new ArrayList<>(2);
        pairList.add(word.get(i));
        pairList.add(word.get(i + 1));
        if (pairs.containsKey(pairList)) {
          pairs.put(pairList, pairs.get(pairList) + wordEntry.getValue());
        } else {
          pairs.put(pairList, wordEntry.getValue());
        }
      }
    }
    return pairs;
  }

  private void mergeVocab(List<List<Short>> pair) {
    List<Short> mergedPair = getMergedPair(pair);
    Map<List<List<Short>>, Integer> newVocab = new HashMap<>(vocab.size());
    for (Entry<List<List<Short>>, Integer> wordEntry : this.vocab.entrySet()) {
      List<List<Short>> word = mergeWord(wordEntry.getKey(), pair, mergedPair);
      newVocab.put(word, wordEntry.getValue());
    }
    vocab = newVocab;
  }

  private static List<Short> getMergedPair(List<List<Short>> pair) {
    List<Short> mergedPair = new ArrayList<>(pair.get(0).size() + pair.get(1).size());
    mergedPair.addAll(pair.get(0));
    mergedPair.addAll(pair.get(1));
    return mergedPair;
  }

  private static List<List<Short>> mergeWord(List<List<Short>> word, List<List<Short>> pair, List<Short> mergedPair) {
    for (int i = 0; i < word.size() - 1; i++) {
      if (word.get(i).equals(pair.get(0)) && word.get(i + 1).equals(pair.get(1))) {
        word.set(i, mergedPair);
        word.remove(i + 1);
      }
    }
    return word;
  }

  @Override
  public short[][] transform(short[] word) {
    // apply merges
    List<List<Short>> wordList = this.getWordList(word);
    for (List<List<Short>> pair : merges) {
      List<Short> mergedPair = getMergedPair(pair);
      wordList = mergeWord(wordList, pair, mergedPair);
    }
    // return merged subwords
    List<short[]> subwords = new ArrayList<>();
    for (List<Short> subword : wordList) {
      if (subword.size() > 1) {
        if (this.hasPositionalAlphabets()) {
          subwords.add(this.shortListToArrayPattern(subword, word.length));
        } else {
          subwords.add(shortListToArray(subword));
        }
      }
    }
    return subwords.toArray(new short[subwords.size()][]);
  }

  private static short[] shortListToArray(List<Short> word) {
    short[] wordArray = new short[word.size()];
    for (int i = 0; i < word.size(); i++) {
      wordArray[i] = word.get(i);
    }
    return wordArray;
  }

  private short[] shortListToArrayPattern(List<Short> word, int length) {
    short[] pattern = new short[length];
    Arrays.fill(pattern, this.getFillCharacter());
    int alphabetSize = this.getInputAlphabetSize();
    for (short s : word) {
      int position = s / alphabetSize;
      short character = (short) (s % alphabetSize);
      pattern[position] = character;
    }
    return pattern;
  }

  private List<List<Short>> getWordList(short[] word) {
    List<List<Short>> wordList = new ArrayList<>(word.length);
    for (int j = 0; j < word.length; j++) {
      List<Short> charList = new ArrayList<>(1);
      short character;
      if (this.hasPositionalAlphabets()) {
        character = (short) (word[j] + this.getInputAlphabetSize() * j);
      } else {
        character = word[j];
      }
      charList.add(character);
      wordList.add(charList);
    }
    return wordList;
  }

}
