package subwordTransformer.bpe;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import subwordTransformer.SupervisedTransformer;
import subwordTransformer.SupervisedUtils;

/**
 * A transformer that uses byte pair encoding to find long representative
 * subsequences.
 */
public class SupervisedBPETransformer extends SupervisedTransformer<BPEParameter> {

  private List<Map<List<List<Short>>, Integer>> vocabs;
  private List<List<List<Short>>> merges;

  private List<List<Short>> bestPair;
  private double bestSigma;

  /**
   * @param alphabetSize        the alphabet size of the input words
   * @param positionalAlphabets whether the transformer should use positional
   *                            alphabets, i.e. different positions in words have
   *                            different meanings
   */
  public SupervisedBPETransformer(int alphabetSize, boolean positionalAlphabets) {
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
  public SupervisedBPETransformer(int alphabetSize, boolean positionalAlphabets, short fillCharacter) {
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
    vocabs = new ArrayList<>();
    for (short[][] classWords : this.getWords()) {
      vocabs.add(BPEUtils.buildVocab(classWords, this.hasPositionalAlphabets(), this.getInputAlphabetSize()));
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
    int numClasses = vocabs.size();
    int[] classSizes = new int[numClasses];
    for (int i = 0; i < numClasses; i++) {
      classSizes[i] = this.getWords()[i].length;
    }
    double max = Math.sqrt((numClasses - 1) / Math.pow(numClasses, 2));
    while (true) {
      if (!continueSearch) {
        List<Map<List<List<Short>>, Integer>> pairsPerClass = this.getStats();
        // find most frequent pair
        bestPair = null;
        bestSigma = -1;

        Set<List<List<Short>>> seenPairs = new HashSet<>();
        for (int i = 0; i < numClasses; i++) {
          for (List<List<Short>> pair : pairsPerClass.get(i).keySet()) {
            if (!seenPairs.contains(pair)) {
              seenPairs.add(pair);
              int[] counts = new int[numClasses];
              for (int j = 0; j < numClasses; j++) {
                Integer count = pairsPerClass.get(j).get(pair);
                counts[j] = count == null ? 0 : count.intValue();
              }
              double sigma = SupervisedUtils.sigma(counts, classSizes, max);
              if (sigma > bestSigma) {
                bestPair = pair;
                bestSigma = sigma;
              }
            }
          }
        }

      } else {
        continueSearch = false;
      }
      if (bestSigma >= this.getParameter().getMinSupport()) {
        merges.add(bestPair);
        this.mergeVocab(bestPair);
      } else {
        break;
      }
    }
  }

  private List<Map<List<List<Short>>, Integer>> getStats() {
    List<Map<List<List<Short>>, Integer>> stats = new ArrayList<>();
    for (Map<List<List<Short>>, Integer> vocab : vocabs) {
      stats.add(BPEUtils.getStats(vocab));
    }
    return stats;
  }

  private void mergeVocab(List<List<Short>> pair) {
    List<Short> mergedPair = BPEUtils.getMergedPair(pair);
    List<Map<List<List<Short>>, Integer>> newVocabs = new ArrayList<>();
    for (Map<List<List<Short>>, Integer> vocab : vocabs) {
      newVocabs.add(BPEUtils.mergeVocab(vocab, pair, mergedPair));
    }
    vocabs = newVocabs;
  }

  @Override
  public short[][] transform(short[] word) {
    return BPEUtils.transform(word, merges, this.hasPositionalAlphabets(), this.getInputAlphabetSize(), this.getFillCharacter());
  }

}
